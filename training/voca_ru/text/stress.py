"""Stress marks: the RUAccent dictionary + nn_accent model (MIT), plus a fallback.

This is the runtime subset the contract ships (contract §9): the 3.19M-form
dictionary, the 19,741-entry omograph list and the small OOV accent model.
The full ``RUAccent().load()`` also pulls the stress-usage predictor, the
omograph transformer and koziev morphology — those are deliberately not used.

Output is the text with combining acute U+0301 after the stressed vowel, the
exact string stored in the manifest as ``norm_text`` (contract §5).
"""

import gzip
import json
import os
import re

from .symbols_ru import STRESS, mark_vowel, strip_stress

VOWEL_LETTERS = "аеёиоуыэюя"
WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)*")
MARKED_WORD_RE = re.compile(r"[а-яё\u0301]+(?:-[а-яё\u0301]+)*")

# Monosyllabic function words are unstressed clitics; RUAccent leaves them
# alone and forcing a mark there would teach the teacher the wrong prosody.
CLITICS = {
    "в", "во", "на", "до", "от", "из", "к", "ко", "по", "с", "со", "у",
    "о", "об", "обо", "за", "под", "над", "при", "для", "без", "через",
    "между", "а", "и", "но", "или", "не", "ни", "же", "ли", "бы", "что",
    "ведь", "вот", "да",
}

# RUAccent's dictionary stores the statistically dominant reading, which is
# wrong for exactly the cases the normaliser generates: "году" after a year is
# prepositional (год+у), "года" after a year/cardinal is genitive (г+ода).
CONTEXT_OVERRIDES = {
    "году": "год+у",
    "года": "г+ода",
}

# Suffixes whose stress is fixed often enough to beat "penultimate".
# Value: 0 = last vowel, 1 = one before last.
FALLBACK_SUFFIXES = [
    ("изация", 2), ("фикация", 2), ("ация", 2),
    ("ование", 2), ("ение", 2), ("ание", 2),
    ("ический", 2), ("ическая", 2), ("ическое", 2),
    ("ировать", 2), ("овать", 0), ("евать", 0),
    ("ативный", 1), ("ография", 2), ("ология", 2),
    ("изм", 0), ("ист", 0),
]


def _vowel_indices(word):
    return [i for i, char in enumerate(word) if char.lower() in VOWEL_LETTERS]


def fallback_stress(word):
    """Uncertain stress for a word no dictionary/model covered.

    ё wins (inherently stressed); then a small suffix table; then the
    penultimate vowel. This is a fallback, not a claim of correctness.
    """
    indices = _vowel_indices(word)
    if not indices:
        return word
    for i, char in enumerate(word):
        if char.lower() == "ё":
            return word[: i + 1] + STRESS + word[i + 1:]
    lower = word.lower()
    for suffix, back in FALLBACK_SUFFIXES:
        if lower.endswith(suffix):
            return mark_vowel(word, max(0, len(indices) - 1 - back))
    return mark_vowel(word, max(0, len(indices) - 2) if len(indices) > 1 else 0)


def _plus_to_acute(text):
    return re.sub(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])", r"\1" + STRESS, text)


def _package_data_dir():
    try:
        import ruaccent

        return os.path.join(os.path.dirname(ruaccent.__file__))
    except Exception:
        return None


def _default_dict_dir():
    base = os.environ.get("VOCA_RUACCENT_WORKDIR")
    if not base:
        cache = os.environ.get("VOCA_CACHE")
        if cache:
            base = os.path.join(cache, "models", "ruaccent")
    if base and os.path.isdir(os.path.join(base, "dictionary")):
        return os.path.join(base, "dictionary")
    package = _package_data_dir()
    if package:
        candidate = os.path.join(package, "dictionary")
        if os.path.isdir(candidate):
            return candidate
    return None


class Stressor:
    """Dictionary-first stress with an ONNX model for OOV words.

    ``load()`` is lazy and idempotent; an unloaded Stressor uses the rule
    fallback only, which is what ``--no-ruaccent`` in the CLI selects.
    """

    def __init__(self, workdir=None, use_model=True):
        self.workdir = workdir
        self.use_model = use_model
        self.accents = None
        self.omographs = {}
        self.yo_words = {}
        self.yo_homographs = {}
        self.model = None
        self.omograph_model = None
        self._chosen = {}
        self._last_oov = False
        self.stats = {"dictionary": 0, "omograph": 0, "model": 0, "fallback": 0}

    def load(self):
        if self.accents is not None:
            return self
        directory = self.workdir or _default_dict_dir()
        if not directory:
            raise FileNotFoundError(
                "RUAccent dictionary not found; set VOCA_RUACCENT_WORKDIR to "
                "the directory that contains dictionary/accents.json.gz")
        with gzip.open(os.path.join(directory, "accents.json.gz")) as handle:
            self.accents = json.load(handle)
        for name in ("omographs", "yo_words", "yo_homographs"):
            path = os.path.join(directory, f"{name}.json.gz")
            if os.path.exists(path):
                with gzip.open(path) as handle:
                    setattr(self, name, json.load(handle))
        if self.use_model:
            package = _package_data_dir()
            nn_dir = os.path.join(package, "nn", "nn_accent") if package else None
            if nn_dir and os.path.exists(os.path.join(nn_dir, "model.onnx")):
                from ruaccent.accent_model import AccentModel

                self.model = AccentModel()
                self.model.load(nn_dir)
            # The omograph classifier reads the whole sentence and picks one of the
            # two readings the dictionary knows for a form like замок. Without it
            # homographs are stuck on whichever reading accents.json happens to hold.
            base = os.path.dirname(directory)
            omograph_dir = os.path.join(base, "nn", "nn_omograph", "tiny")
            if os.path.exists(os.path.join(omograph_dir, "model.onnx")):
                from ruaccent.omograph_model import OmographModel

                self.omograph_model = OmographModel()
                self.omograph_model.load(omograph_dir)
            else:
                self.omograph_model = None
        return self

    def _resolve_omographs(self, text):
        """Sentence -> {position: chosen variant} for every homograph in it."""
        if not (self.omographs and self.omograph_model):
            return {}
        words = text.split()
        positions, variants = [], []
        for index, word in enumerate(words):
            plain = re.sub(r"[^а-яёa-z-]", "", word.lower())
            options = self.omographs.get(plain)
            if options and len(options) > 1:
                positions.append(index)
                variants.append(options)
        if not positions:
            return {}
        marked = list(words)
        for index in positions:
            marked[index] = " <w>" + marked[index] + "</w> "
        sentence = " ".join(marked)
        # classify() compares one sentence against one word's readings at a time:
        # it wants the text repeated once per hypothesis of that single word.
        resolved = {}
        for index, options in zip(positions, variants):
            try:
                resolved[index] = self.omograph_model.classify(
                    [sentence] * len(options), list(options), [len(options)]
                )[0]
            except Exception:
                resolved[index] = options[0]
        self.stats["omograph"] = self.stats.get("omograph", 0) + len(positions)
        return resolved

    def _model_accent(self, word):
        import numpy as np

        inputs = self.model.tokenizer(word.lower(), return_tensors="np")
        inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        wanted = {item.name for item in self.model.session.get_inputs()}
        if "token_type_ids" in wanted and "token_type_ids" not in inputs:
            inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
        outputs = self.model.session.run(None, inputs)
        names = {item.name: i for i, item in enumerate(self.model.session.get_outputs())}
        logits = outputs[names["logits"]]
        scores = np.max(logits, axis=-1)[0]
        labels = np.argmax(logits, axis=-1)[0]
        predictions = [{"label": self.model.id2label[str(label)], "score": float(score)}
                       for label, score in zip(labels, scores)]
        return self.model.render_stress(word, predictions)

    def _word(self, word):
        """One lowercase word -> marked word, updating stats."""
        if STRESS in word:
            # the user marked this word by hand; that outranks every table
            self.stats["manual"] = self.stats.get("manual", 0) + 1
            return word
        if word in CONTEXT_OVERRIDES:
            self.stats["omograph"] += 1
            return _plus_to_acute(CONTEXT_OVERRIDES[word])
        if word in self.yo_words:
            word = self.yo_words[word]
        # A homograph resolved by the sentence's context outranks the single
        # reading accents.json stores for the same form.
        if self._chosen.get(word):
            return _plus_to_acute(self._chosen[word])
        marked = self.accents.get(word)
        if marked is not None:
            self.stats["dictionary"] += 1
            return _plus_to_acute(marked)
        variants = self.omographs.get(word)
        if variants:
            self.stats["omograph"] += 1
            return _plus_to_acute(variants[0])
        if self.model is not None:
            marked = self._model_accent(word)
            if "+" in marked:
                self.stats["model"] += 1
                return _plus_to_acute(marked)
        self.stats["fallback"] += 1
        return fallback_stress(word)

    def _mark_word(self, word):
        if "-" in word:
            return "-".join(self._word(part) for part in word.split("-"))
        return self._word(word)

    def stressed(self, text):
        """Lowercase normalised text -> (stressed text, oov flag).

        Acute marks already present are the user's own choice and survive.
        """
        text = _plus_to_acute(text)  # "+" before a vowel is a hand-written mark
        if self.accents is None:
            self._last_oov = True
            return _ensure_one_mark(text), True
        oov = False
        self.stats = {"dictionary": 0, "omograph": 0, "model": 0, "fallback": 0}
        resolved = self._resolve_omographs(text)
        # key by the plain form so _word can look it up after stripping punctuation
        words = text.split()
        self._chosen = {}
        for index, variant in resolved.items():
            plain = re.sub(r"[^а-яёa-z-]", "", words[index].lower())
            self._chosen.setdefault(plain, variant)

        def repl(match):
            nonlocal oov
            word = match.group(0)
            for part in word.split("-"):
                if STRESS in part:
                    continue
                if len(part) > 1 and part not in self.accents \
                        and part not in self.omographs and part not in CONTEXT_OVERRIDES:
                    oov = True
            if word.lower() in CLITICS:
                return word
            return self._mark_word(word)

        marked = MARKED_WORD_RE.sub(repl, text)
        self._last_oov = oov
        return _ensure_one_mark(marked), oov

    @property
    def oov(self):
        return self._last_oov


def _ensure_one_mark(text):
    """Every word with a vowel gets exactly one mark."""
    def repl(match):
        word = match.group(0)
        if STRESS in word or "+" in word:
            return _plus_to_acute(word)
        if word.lower() in CLITICS:
            return word
        return fallback_stress(word)

    return MARKED_WORD_RE.sub(repl, text)


def stress_text(text, stressor=None):
    if stressor is None:
        stressor = _DEFAULT
    return stressor.stressed(text)


_DEFAULT = Stressor()


def words_with_marks(text):
    """[(word, vowel_index) or (word, None)] for tests and reports."""
    out = []
    for match in MARKED_WORD_RE.finditer(text):
        word = match.group(0)
        index = None
        for i, char in enumerate(word):
            if char == STRESS:
                index = sum(1 for c in word[:i] if c.lower() in VOWEL_LETTERS) - 1
                break
        out.append((strip_stress(word), index))
    return out

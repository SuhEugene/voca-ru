"""Self-checks for the eSpeak-ru parsing and the label map.

    python scripts/align_spike/test_mapping.py

No test framework: asserts are enough for a measurement script, and this keeps
the spike dependency-free. The vocabulary check is skipped when the model is not
in the shared cache.
"""

import json
import os
from pathlib import Path

import mapping
from espeak_ru import parse_espeak

# Every phone eSpeak 1.52 `ru` produced over the 775 FLEURS test transcripts.
INVENTORY = ("ʌ i n t s ɪ ɑ o r y v j k a e p ɭ u m ɭʲ tʲ nʲ d rʲ b sʲ z x mʲ ɡ "
             "tʃʲ ts f ja dʲ vʲ ɛ ʃ ʒ kʲ ju ɕ pʲ ɡʲ ə bʲ ʑ u\" ɵ fʲ ɔ iː ɪ^ eɪ aɪ "
             "əʊ ɹ tʃ dʒ ɑː θ l eə ɒ h əl ɜː dʒʲ uː w ø").split()


def test_parse():
    words = parse_espeak("p_ʌ_mʲ_ˈi_m_ʌ s__rʲ_ˈe_d_u\n")
    assert words[0] == [("p", 0), ("ʌ", 0), ("mʲ", 0), ("i", 1), ("m", 0), ("ʌ", 0)], words[0]
    assert words[1][1] == ("rʲ", 0) and words[1][2] == ("e", 1), words[1]  # empty unit dropped
    assert isinstance(words[0][1][0], str)


def test_language_tags():
    words = parse_espeak("(en)_t_ˈɛ_s_t_(ru)__ (en)_ˈɛ_n_(ru)")
    phones = [p for w in words for p, _ in w]
    assert "en" not in phones and "ru" not in phones, phones


def test_every_inventory_phone_has_a_label():
    vocab = None
    model = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).parent / "models/wav2vec2-lv-60-espeak-cv-ft/vocab.json"
    if model.exists():
        vocab = json.loads(model.read_text(encoding="utf-8"))
    for phone in INVENTORY:
        target = mapping.CANONICAL.get(phone) or phone
        if vocab is not None:
            if target in vocab:
                continue
            target = mapping.FALLBACK.get(phone)
            assert target is not None or phone in mapping.EXPAND, f"{phone!r} has no label"
            assert target in vocab, f"{phone!r} -> {target!r} is not in vocab.json"
        else:
            assert target or phone in mapping.EXPAND, phone


def test_tables_are_consistent():
    for phone, target in mapping.CANONICAL.items():
        assert target and target != phone, phone
    for phone, target in mapping.FALLBACK.items():
        assert target and target != phone, phone
    for phone, parts in mapping.EXPAND.items():
        assert len(parts) == 2 and phone not in parts, phone


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok", name)

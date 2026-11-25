// Stress marks: the shipped RUAccent-derived dictionary plus the Python
// frontend's rule fallback. Port of training/voca_ru/text/stress.py, minus the
// ONNX model that is not shipped in the GGUF.

const VOWELS = "аеёиоуыэюя";
const ACUTE = "\u0301";

// Monosyllabic function words are unstressed clitics.
const CLITICS = new Set(["в", "во", "на", "до", "от", "из", "к", "ко", "по", "с", "со", "у",
  "о", "об", "обо", "за", "под", "над", "при", "для", "без", "через",
  "между", "а", "и", "но", "или", "не", "ни", "же", "ли", "бы", "что",
  "ведь", "вот", "да"]);

// The dictionary's dominant reading is wrong for years.
const CONTEXT_OVERRIDES = { "году": "год+у", "года": "г+ода" };

const FALLBACK_SUFFIXES = [
  ["изация", 2], ["фикация", 2], ["ация", 2], ["ование", 2], ["ение", 2], ["ание", 2],
  ["ический", 2], ["ическая", 2], ["ическое", 2], ["ировать", 2], ["овать", 0],
  ["евать", 0], ["ативный", 1], ["ография", 2], ["ология", 2], ["изм", 0], ["ист", 0],
];

const WORD_RE = /[а-яё]+(?:-[а-яё]+)*/g;
const MARKED_WORD_RE = /[а-яё\u0301]+(?:-[а-яё\u0301]+)*/g;

const isVowel = (char) => VOWELS.includes(char.toLowerCase());

export function plusToAcute(text) {
  return text.replace(/\+([аеёиоуыэюя])/g, "$1" + ACUTE);
}

export function stripStress(text) {
  return text.replaceAll(ACUTE, "");
}

const vowelIndices = (word) => [...word].map((c, i) => (isVowel(c) ? i : -1)).filter((i) => i >= 0);

export function markVowel(word, index) {
  let seen = -1;
  for (let i = 0; i < word.length; i++) {
    if (!isVowel(word[i])) continue;
    if (++seen === index) return word.slice(0, i + 1) + ACUTE + word.slice(i + 1);
  }
  return word;
}

export function fallbackStress(word) {
  const indices = vowelIndices(word);
  if (!indices.length) return word;
  const yo = word.toLowerCase().indexOf("ё");
  if (yo >= 0) return word.slice(0, yo + 1) + ACUTE + word.slice(yo + 1);
  const lower = word.toLowerCase();
  for (const [suffix, back] of FALLBACK_SUFFIXES) {
    if (lower.endsWith(suffix)) return markVowel(word, Math.max(0, indices.length - 1 - back));
  }
  return markVowel(word, indices.length > 1 ? indices.length - 2 : 0);
}

export class Stressor {
  constructor(dictionaryText) {
    this.dictionary = new Map();
    for (const line of dictionaryText.split("\n")) {
      const at = line.indexOf("\t");
      if (at > 0) this.dictionary.set(line.slice(0, at), line.slice(at + 1));
    }
  }

  part(word) {
    if (word in CONTEXT_OVERRIDES) return plusToAcute(CONTEXT_OVERRIDES[word]);
    const marked = this.dictionary.get(word);
    return marked === undefined ? fallbackStress(word) : plusToAcute(marked);
  }

  word(word) {
    const parts = word.split("-");
    return parts.length > 1 ? parts.map((part) => this.part(part)).join("-") : this.part(word);
  }

  // Lowercase normalised text -> { text with acute marks, oov flag }.
  stressed(text) {
    text = stripStress(text);
    let oov = false;
    const marked = text.replace(WORD_RE, (word) => {
      for (const part of word.split("-")) {
        if (part.length > 1 && !this.dictionary.has(part) && !(part in CONTEXT_OVERRIDES)) oov = true;
      }
      return CLITICS.has(word) ? word : this.word(word);
    });
    const ensured = marked.replace(MARKED_WORD_RE, (word) => {
      if (word.includes(ACUTE) || word.includes("+")) return plusToAcute(word);
      return CLITICS.has(word.toLowerCase()) ? word : fallbackStress(word);
    });
    return { stressed: ensured, oov };
  }
}

export { isVowel, ACUTE };

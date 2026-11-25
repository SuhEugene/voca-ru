// Russian G2P: the eSpeak-derived pronunciation lexicon from the GGUF plus the
// letter rules of crates/voca-ru-text/src/g2p.rs. Mirrors
// training/voca_ru/text/g2p.py tokenization.

import { ACUTE, stripStress } from "./stress.js";

const VOWEL_LETTERS = "аеёиоуыэюя";
const VOWEL_PHONES = ["ʌ", "a", "ɑ", "ɪ", "i", "ɛ", "e", "ə", "o", "u", "y", "ɵ"];

const HARD = {
  б: "b", в: "v", г: "ɡ", д: "d", ж: "ʒ", з: "z", й: "j", к: "k", л: "ɭ", м: "m",
  н: "n", п: "p", р: "r", с: "s", т: "t", ф: "f", х: "x", ц: "ts", ч: "tʃʲ", ш: "ʃ", щ: "ɕ",
};
const SOFT = {
  б: "bʲ", в: "vʲ", г: "ɡʲ", д: "dʲ", з: "ʑ", к: "kʲ", л: "ɭʲ", м: "mʲ", н: "nʲ",
  п: "pʲ", р: "rʲ", с: "sʲ", т: "tʲ", ф: "fʲ",
};
const SOFT_VOWELS = new Set(["е", "ё", "и", "ю", "я"]);

const isVowelLetter = (char) => VOWEL_LETTERS.includes(char);
const isVowelPhone = (phone) => VOWEL_PHONES.includes(phone);

const SP = "<sp>", BOS = "<bos>", EOS = "<eos>";
const PHONE_SET = new Set([...Object.values(HARD), ...Object.values(SOFT), ...VOWEL_PHONES]);
export const isPhone = (symbol) => PHONE_SET.has(symbol);

function targetVowelIndex(word) {
  let count = 0;
  for (const char of word) {
    if (char === ACUTE) return Math.max(0, count - 1);
    if (isVowelLetter(char)) count++;
  }
  return null;
}

function applyStress(phones, target) {
  if (target == null) return phones;
  const slots = phones.flatMap(([phone], i) => (isVowelPhone(phone) ? [i] : []));
  if (target >= slots.length) return phones;
  for (const phone of phones) phone[1] = 0;
  phones[slots[target]][1] = 1;
  return phones;
}

function ruleWord(word) {
  const plain = [...stripStress(word)];
  if (!plain.length) return [];
  const vowelPositions = plain.flatMap((char, i) => (isVowelLetter(char) ? [i] : []));
  const stressIndex = Math.min(targetVowelIndex(word) ?? 0, Math.max(0, vowelPositions.length - 1));
  const phones = [];
  let previousVowel = false, previousSoftSign = false, vowelSeen = 0;

  for (let i = 0; i < plain.length; i++) {
    const char = plain[i];
    if (char === "ь" || char === "ъ") {
      previousSoftSign = char === "ь";
      previousVowel = false;
      continue;
    }
    if (HARD[char]) {
      const next = plain[i + 1] || " ";
      const soft = SOFT_VOWELS.has(next) || next === "ь";
      phones.push([(soft && SOFT[char]) || HARD[char], 0]);
      previousSoftSign = false;
      previousVowel = false;
      continue;
    }
    if (!isVowelLetter(char)) continue;
    const stressed = vowelSeen === stressIndex;
    const position = vowelPositions.indexOf(i);
    const pretonic = stressIndex > 0 && position + 1 === stressIndex;
    const soft = previousSoftSign;
    const glide = ["е", "ё", "ю", "я"].includes(char) && (i === 0 || previousVowel || previousSoftSign);
    if (glide) phones.push(["j", 0]);
    let phone;
    if (char === "а") phone = stressed ? (soft ? "a" : "ɑ") : position + 1 === plain.length ? "a" : pretonic && !soft ? "ʌ" : "ə";
    else if (char === "о") phone = stressed ? "o" : pretonic ? "ʌ" : "ə";
    else if (char === "у") phone = "u";
    else if (char === "ы") phone = "y";
    else if (char === "э") phone = "ɛ";
    else if (char === "е") phone = stressed ? "e" : "ɪ";
    else if (char === "и") phone = stressed ? "i" : "ɪ";
    else if (char === "ё") phone = "ɵ";
    else if (char === "ю") phone = "u";
    else if (char === "я") phone = stressed ? "a" : "ɪ";
    else continue;
    phones.push([phone, stressed ? 1 : 0]);
    previousVowel = true;
    previousSoftSign = false;
    vowelSeen++;
  }
  return phones;
}

export class G2P {
  constructor(lexiconText, symbols, ids) {
    this.symbols = symbols;
    this.ids = ids;
    this.known = new Set(symbols);
    this.lexicon = new Map();
    for (const line of lexiconText.split("\n")) {
      const at = line.indexOf("\t");
      if (at <= 0) continue;
      const phones = line.slice(at + 1).split(/\s+/).filter(Boolean).flatMap((token) => {
        const cut = token.lastIndexOf(":");
        const name = cut > 0 ? token.slice(0, cut) : token;
        if (!this.known.has(name)) return [];
        return [[name, cut > 0 ? Number(token.slice(cut + 1)) || 0 : 0]];
      });
      this.lexicon.set(line.slice(0, at), phones);
    }
  }

  wordPhones(word) {
    const target = targetVowelIndex(word);
    const plain = stripStress(word);
    if (this.lexicon.has(plain)) return applyStress(this.lexicon.get(plain).map((p) => [...p]), target);
    if (plain.includes("-")) {
      const out = [];
      for (const part of plain.split("-")) {
        if (out.length) out.push([SP, 0]);
        out.push(...this.wordPhones(part));
      }
      return applyStress(out, target);
    }
    return applyStress(ruleWord(word), target);
  }

  pieces(text) {
    const out = [];
    for (const [token] of text.matchAll(/[а-яё\u0301]+(?:-[а-яё\u0301]+)*|[^\sа-яё\u0301]/g)) {
      const letters = [...token].every((char) => /\p{L}/u.test(char) || char === "-" || char === ACUTE);
      if (letters && /\p{L}/u.test(token)) out.push(["word", token]);
      else if (this.known.has(token)) out.push(["punct", token]);
    }
    return out;
  }

  tokenize(text) {
    const symbols = [BOS], stress = [0];
    let hasPhone = false;
    for (const [kind, value] of this.pieces(text)) {
      if (kind === "punct") {
        if (symbols.at(-1) !== value) { symbols.push(value); stress.push(0); }
        continue;
      }
      if (isPhone(symbols.at(-1)) || symbols.at(-1) === "<unk>") { symbols.push(SP); stress.push(0); }
      for (const [phone, level] of this.wordPhones(value)) {
        if (phone === SP) {
          if (symbols.at(-1) !== SP) { symbols.push(SP); stress.push(0); }
          continue;
        }
        if (isPhone(phone)) hasPhone = true;
        symbols.push(phone);
        stress.push(level);
      }
    }
    symbols.push(EOS);
    stress.push(0);
    if (!hasPhone) throw new Error("the text has nothing to pronounce");
    return { symbols, ids: symbols.map((symbol) => this.ids.get(symbol)), stress };
  }
}

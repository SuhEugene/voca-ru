// Russian text normalisation, a direct port of
// training/voca_ru/text/normalize.py: numbers, dates, times, money, ordinals,
// units, abbreviations and Latin runs become lowercase Russian words; „ё“ is
// kept; punctuation is mapped onto the 76-symbol inventory.

import * as numbers from "./numbers.js";

const MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня",
                    "июля", "августа", "сентября", "октября", "ноября", "декабря"];
const MONTH_RE = `(?:${MONTHS_GEN.join("|")})`;

// JS \b is ASCII-only, Python's \b counts Cyrillic as word characters.
const WB = String.raw`(?:(?<=[\p{L}\p{N}_])(?![\p{L}\p{N}_])|(?<![\p{L}\p{N}_])(?=[\p{L}\p{N}_]))`;
const WORD = (pattern) => new RegExp(pattern, "gu");

const LATIN_NAMES = {
  a: "эй", b: "би", c: "си", d: "ди", e: "и", f: "эф", g: "джи", h: "эйч", i: "ай",
  j: "джей", k: "кей", l: "эл", m: "эм", n: "эн", o: "о", p: "пи", q: "кью", r: "ар",
  s: "эс", t: "ти", u: "ю", v: "ви", w: "дабл ю", x: "экс", y: "уай", z: "зет",
};

const UNITS = {
  км: ["километр", "километра", "километров"],
  м: ["метр", "метра", "метров"],
  см: ["сантиметр", "сантиметра", "сантиметров"],
  мм: ["миллиметр", "миллиметра", "миллиметров"],
  кг: ["килограмм", "килограмма", "килограммов"],
  г: ["грамм", "грамма", "граммов"],
  мг: ["миллиграмм", "миллиграмма", "миллиграммов"],
  л: ["литр", "литра", "литров"],
  мл: ["миллилитр", "миллилитра", "миллилитров"],
  га: ["гектар", "гектара", "гектаров"],
  руб: ["рубль", "рубля", "рублей"],
  коп: ["копейка", "копейки", "копеек"],
};

const DIGIT_WORDS = ["ноль", "один", "два", "три", "четыре",
                     "пять", "шесть", "семь", "восемь", "девять"];

const GEN_PREPS = new Set(["до", "от", "из", "без", "около", "для", "у", "после",
                           "кроме", "ради", "больше", "меньше", "свыше", "порядка"]);
const DAT_PREPS = new Set(["к", "по"]);
const PREP_PREPS = new Set(["в", "на", "о", "об", "при"]);
const INS_PREPS = new Set(["с", "между", "над", "под", "перед"]);

function clean(text) {
  return text.normalize("NFC")
    .replace(/[\u00a0\u202f\u2009]/g, " ")
    .replace(/\u0301/g, "")
    .toLowerCase()
    .replace(/[\u200b\u200c\u200d\ufeff\t\r\n]/g, " ");
}

const mergeThousands = (text) =>
  text.replace(/(?<!\d)(\d{1,3})(?:[ ]\d{3})+(?!\d)/g, (all) => all.replace(/ /g, ""));

const SCALE_UNITS = {
  "тыс.": ["тысяча", "тысячи", "тысяч", numbers.F],
  "тыс": ["тысяча", "тысячи", "тысяч", numbers.F],
  "млн": ["миллион", "миллиона", "миллионов", numbers.M],
  "млрд": ["миллиард", "миллиарда", "миллиардов", numbers.M],
};

function expandScales(text) {
  return text.replace(WORD(String.raw`(?<!\d)(\d{1,9})\s*(тыс\.|тыс${WB}|млн${WB}|млрд${WB})`), (all, digits, unit) => {
    const value = parseInt(digits, 10);
    const [one, few, many, gender] = SCALE_UNITS[unit];
    return [...numbers.cardinal(value, numbers.NOM, gender), numbers.agree(value, one, few, many)].join(" ");
  });
}

function expandAbbreviations(text) {
  const rules = [
    [String.raw`т\.\s*д\.`, "так далее"],
    [String.raw`т\.\s*е\.`, "то есть"],
    [String.raw`т\.\s*к\.`, "так как"],
    [String.raw`т\.\s*п\.`, "тому подобное"],
    [String.raw`и\s+т\.\s*д\.`, "и так далее"],
    [String.raw`гг\.`, "годы"],
    [String.raw`см\.`, "смотри"],
    [String.raw`стр\.`, "страница"],
    [String.raw`ул\.`, "улица"],
    [String.raw`пр\.`, "проспект"],
    [String.raw`кв\.`, "квартира"],
    [String.raw`им\.`, "имени"],
    [String.raw`проф\.`, "профессор"],
    [String.raw`тов\.`, "товарищ"],
    [String.raw`тел\.`, "телефон"],
    [String.raw`рис\.`, "рисунок"],
    [String.raw`табл\.`, "таблица"],
    [String.raw`тыс\.`, "тысяч"],
    [String.raw`млн`, "миллион"],
    [String.raw`млрд`, "миллиард"],
    [String.raw`руб\.`, "рублей"],
    [String.raw`коп\.`, "копеек"],
    [String.raw`др\.`, "другое"],
    [String.raw`№`, "номер"],
  ];
  let out = text;
  for (const [pattern, replacement] of rules) out = out.replace(WORD(pattern), replacement);
  out = out.replace(WORD(String.raw`(?<=\d)\s+г\.`), " года");
  out = out.replace(WORD(String.raw`(?<=\d)\s+с\.`), " страница");
  out = out.replace(WORD(String.raw`(?<=\d)\s+д\.`), " дом");
  out = out.replace(WORD(`${WB}г\\.`), "город");
  out = out.replace(WORD(`${WB}с\\.`), "страница");
  out = out.replace(WORD(`${WB}д\\.`), "дом");
  return out;
}

const MASCULINE_GENITIVES = new Set(["часа", "дня", "года", "месяца", "рубля", "метра",
  "километра", "литра", "процента", "килограмма", "грамма", "доллара", "миллиона"]);

const TRIM = /^[.,!?;:«»"'()[\]{}—–…]+|[.,!?;:«»"'()[\]{}—–…]+$/g;
const trimWord = (word) => word.replace(TRIM, "");

function genderOf(word) {
  word = trimWord(word);
  if (!word) return numbers.M;
  if (MASCULINE_GENITIVES.has(word)) return numbers.M;
  if (/(ая|яя)$/.test(word)) return numbers.F;
  if (/(ое|ее|мя)$/.test(word)) return numbers.N;
  if (/[аяыи]$/.test(word)) return numbers.F;
  if (/[ое]$/.test(word)) return numbers.N;
  return numbers.M;
}

function nearestWordBefore(text, pos) {
  const head = text.slice(0, pos).replace(/\s+$/, "");
  if (!head) return "";
  return trimWord(head.split(/\s+/).pop()).toLowerCase();
}

function wordAfter(text, pos) {
  const rest = text.slice(pos).replace(/^\s+/, "");
  const match = rest.match(/^[а-яё-]+/);
  return match ? match[0] : "";
}

function caseFor(text, pos, lookahead = true, after = null) {
  const prep = nearestWordBefore(text, pos);
  if (GEN_PREPS.has(prep)) return numbers.GEN;
  if (DAT_PREPS.has(prep)) return numbers.DAT;
  if (INS_PREPS.has(prep)) return numbers.INS;
  if (PREP_PREPS.has(prep)) {
    const next = lookahead ? wordAfter(text, after == null ? pos : after) : "";
    if ((prep === "в" || prep === "на") && !/(ах|ях|е|и)$/.test(next)) return numbers.ACC;
    return numbers.PREP;
  }
  return numbers.NOM;
}

const dayOrdinal = (day) => numbers.ordinal(parseInt(day, 10), numbers.GEN, numbers.M);
const yearOrdinal = (year, c) => numbers.ordinal(parseInt(year, 10), c, numbers.M);

function expandDates(text) {
  let out = text.replace(WORD(String.raw`(?<![\p{L}\p{N}_])(\d{1,2})\.(\d{1,2})\.(\d{4})(?![\p{L}\p{N}_])`), (all, day, month, year) => {
    if (!(Number(month) >= 1 && Number(month) <= 12)) return all;
    return `${dayOrdinal(day)} ${MONTHS_GEN[Number(month) - 1]} ${yearOrdinal(year, numbers.GEN)} года`;
  });
  const pattern = String.raw`(?<![\p{L}\p{N}_])(\d{1,2})\s+(${MONTH_RE})(?:\s+(\d{4})(?:\s+(?:года|году|год))?)?${WB}`;
  out = out.replace(WORD(pattern), (all, day, month, year) =>
    `${dayOrdinal(day)} ${month}` + (year ? ` ${yearOrdinal(year, numbers.GEN)} года` : ""));
  return out;
}

function expandTime(text) {
  return text.replace(WORD(String.raw`(?<![\p{L}\p{N}_])(\d{1,2}):(\d{2})(?![\p{L}\p{N}_])`), (all, hour, minute) => {
    if (Number(hour) > 23 || Number(minute) > 59) return all;
    const words = numbers.cardinal(Number(hour), numbers.NOM, numbers.M);
    if (minute === "00") words.push("ноль", "ноль");
    else if (minute.startsWith("0")) words.push("ноль", ...numbers.cardinal(Number(minute), numbers.NOM, numbers.M));
    else words.push(...numbers.cardinal(Number(minute), numbers.NOM, numbers.M));
    return words.join(" ");
  });
}

function decimalMoney(match, whole, minor, unit, offset, text) {
  const value = parseInt(whole, 10);
  const fraction = parseInt(minor.padEnd(2, "0"), 10);  // ".1" is ten cents
  if (unit === "$" || unit === "€") {
    const words = [...numbers.cardinal(value, numbers.NOM, numbers.M)];
    words.push(unit === "$" ? numbers.agree(value, "доллар", "доллара", "долларов") : "евро");
    if (fraction) {
      words.push(...numbers.cardinal(fraction, numbers.NOM, numbers.M), numbers.agree(fraction, "цент", "цента", "центов"));
    }
    return words.join(" ");
  }
  const c = caseFor(text, offset, false);
  const words = [...numbers.cardinal(value, c, numbers.M), numbers.agree(value, "рубль", "рубля", "рублей")];
  if (fraction) {
    words.push(...numbers.cardinal(fraction, numbers.NOM, numbers.F), numbers.agree(fraction, "копейка", "копейки", "копеек"));
  }
  return words.join(" ");
}

function expandMoney(text) {
  let out = text.replace(new RegExp(String.raw`(?<!\d)(\d{1,9})[.,](\d{1,2})\s*([$€₽])`, "g"),
    (all, whole, minor, unit, offset) => decimalMoney(all, whole, minor, unit, offset, text));
  out = out.replace(new RegExp(String.raw`(?<!\d)(\d{1,9})[.,](\d{1,2})\s*(руб(?:\.|ль|ля|лей)?)(?![а-яё])`, "gu"),
    (all, whole, minor, unit, offset) => decimalMoney(all, whole, minor, unit, offset, text));
  out = out.replace(WORD(String.raw`(?<!\d)(\d{1,9})\s*(?:₽|руб(?:\.|ль|ля|лей)?${WB})`), (all, digits, offset) => {
    const value = parseInt(digits, 10);
    const c = caseFor(text, offset, false);
    return [...numbers.cardinal(value, c, numbers.M), numbers.agree(value, "рубль", "рубля", "рублей")].join(" ");
  });
  out = out.replace(WORD(String.raw`(?<!\d)(\d{1,9})\s*(?:коп\.?|копейка|копейки|копеек)${WB}`), (all, digits, offset) => {
    const value = parseInt(digits, 10);
    const c = caseFor(out, offset, false);
    return [...numbers.cardinal(value, c, numbers.F), numbers.agree(value, "копейка", "копейки", "копеек")].join(" ");
  });
  out = out.replace(WORD(String.raw`(?<!\d)(\d{1,9})\s*€`), (all, digits) =>
    [...numbers.cardinal(parseInt(digits, 10), numbers.NOM, numbers.M), "евро"].join(" "));
  out = out.replace(WORD(String.raw`(?<!\d)(\d{1,9})\s*\$`), (all, digits) => {
    const value = parseInt(digits, 10);
    return [...numbers.cardinal(value, numbers.NOM, numbers.M), numbers.agree(value, "доллар", "доллара", "долларов")].join(" ");
  });
  return out;
}

function expandPercent(text) {
  return text.replace(WORD(String.raw`(?<!\d)(\d{1,9})\s*%`), (all, digits) => {
    const value = parseInt(digits, 10);
    return [...numbers.cardinal(value, numbers.NOM, numbers.M), numbers.agree(value, "процент", "процента", "процентов")].join(" ");
  });
}

const ORDINAL_SUFFIX = {
  "й": [numbers.NOM, numbers.M], "ый": [numbers.NOM, numbers.M], "ой": [numbers.NOM, numbers.M],
  "го": [numbers.GEN, numbers.M], "му": [numbers.DAT, numbers.M], "м": [numbers.INS, numbers.M],
  "х": [numbers.PREP, numbers.M], "я": [numbers.NOM, numbers.F], "ю": [numbers.ACC, numbers.F],
  "е": [numbers.NOM, numbers.N], "ая": [numbers.NOM, numbers.F], "ое": [numbers.NOM, numbers.N],
};

function expandOrdinals(text) {
  return text.replace(WORD(String.raw`(?<!\d)(\d{1,4})-?(й|ый|ой|го|му|м|х|я|ю|е|ая|ое)${WB}`), (all, digits, suffix) => {
    const [c, g] = ORDINAL_SUFFIX[suffix] || [numbers.NOM, numbers.M];
    return numbers.ordinal(parseInt(digits, 10), c, g);
  });
}

function expandYears(text) {
  return text.replace(WORD(String.raw`(?<![\p{L}\p{N}_])(\d{4})\s+(году|года|год)${WB}`), (all, year, suffix) => {
    const c = suffix === "году" ? numbers.PREP : suffix === "года" ? numbers.GEN : numbers.NOM;
    return `${yearOrdinal(year, c)} ${suffix}`;
  });
}

function expandUnits(text) {
  // Same alternation order as Python's sorted(UNITS, key=len, reverse=True).
  const unitRe = ["руб", "коп", "км", "см", "мм", "кг", "мг", "мл", "га", "м", "г", "л"].join("|");
  const pattern = String.raw`(?<!\d)(\d{1,9})\s*(${unitRe})${WB}`;
  return text.replace(WORD(pattern), (all, digits, unit, offset) => {
    const value = parseInt(digits, 10);
    const [one, few, many] = UNITS[unit];
    const c = caseFor(text, offset, false);
    return [...numbers.cardinal(value, c, numbers.M), numbers.agree(value, one, few, many)].join(" ");
  });
}

function expandDecimals(text) {
  return text.replace(WORD(String.raw`(?<!\d)(?:\d{1,3},\d{1,3})(?!\d)`), (all, offset) => {
    const gender = genderOf(wordAfter(text, offset + all.length));
    const c = caseFor(text, offset, false);
    return numbers.decimal(all, c, gender).join(" ");
  });
}

const digitByDigit = (value) => [...value].filter((char) => /\d/.test(char)).map((char) => DIGIT_WORDS[Number(char)]);

function expandPhones(text) {
  return text.replace(WORD(String.raw`(номер[а-яё]*|телефон[а-яё]*|тел)\s+(\d[\d\s-]*\d)`), (all, prefix, rest) => {
    const digits = [...rest].filter((char) => /\d/.test(char)).join("");
    return `${prefix} ${digitByDigit(digits).join(" ")}`;
  });
}

const LONG_CONTEXT = /(?:инн|огрн|снилс|номер|телефон|тел|счёт|счет|билет|платформа)\s*$/;

function expandLongNumbers(text) {
  return text.replace(/(?<!\d)\d+(?!\d)/g, (value, offset) => {
    const before = text.slice(0, offset).toLowerCase().slice(-40);
    if (value.length >= 8 || LONG_CONTEXT.test(before)) return digitByDigit(value).join(" ");
    const c = caseFor(text, offset, true, offset + value.length);
    const gender = genderOf(wordAfter(text, offset + value.length));
    return numbers.cardinal(parseInt(value, 10), c, gender).join(" ");
  });
}

function expandLatin(text) {
  return text.replace(/[a-z]+/g, (run) => [...run].map((char) => LATIN_NAMES[char] || char).join(" "));
}

function punctuation(text) {
  let out = text.replace(/\.{2,}/g, "…")
    .replace(/„|“/g, "«").replace(/”/g, "»")
    .replace(/–/g, "—").replace(/−/g, "-").replace(/’/g, "'").replace(/`/g, "'");
  let open = true;
  out = out.replace(/"/g, () => ((open = !open) ? "»" : "«"));
  const allowed = new Set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя ,.!?;:\"'()—…«»-–№%[]{}");
  return [...out].map((char) => (allowed.has(char) ? char : " ")).join("");
}

function spaces(text) {
  return text.replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?…»)\]])/g, "$1")
    .replace(/([«(\[])\s+/g, "$1")
    .trim();
}

export function normalize(text) {
  let out = clean(text);
  out = expandPhones(out);
  out = mergeThousands(out);
  out = expandScales(out);
  out = expandAbbreviations(out);
  out = expandDates(out);
  out = expandTime(out);
  out = expandYears(out);
  out = expandMoney(out);
  out = expandPercent(out);
  out = expandOrdinals(out);
  out = expandUnits(out);
  out = expandDecimals(out);
  out = expandLongNumbers(out);
  out = expandLatin(out);
  out = punctuation(out);
  return spaces(out);
}

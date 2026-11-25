// Russian cardinal and ordinal numerals, a direct port of
// training/voca_ru/text/numbers.py. Same cases, same tables, same wording.

export const NOM = "nom", GEN = "gen", DAT = "dat", ACC = "acc", INS = "ins", PREP = "prep";
export const M = "m", F = "f", N = "n";
const CASES = [NOM, GEN, DAT, ACC, INS, PREP];
const cidx = (c) => CASES.indexOf(c);

const S = (s) => s.split(" ");
const ZERO = S("ноль ноля нолю ноль нолём ноле");
const ONE = {
  [M]: S("один одного одному один одним одном"),
  [F]: S("одна одной одной одну одной одной"),
  [N]: S("одно одного одному одно одним одном"),
};
const TWO = {
  [M]: S("два двух двум два двумя двух"),
  [F]: S("две двух двум две двумя двух"),
  [N]: S("два двух двум два двумя двух"),
};
const SMALL = {
  3: S("три трёх трём три тремя трёх"),
  4: S("четыре четырёх четырём четыре четырьмя четырёх"),
};
// 5–20, 30, 40…90: (nom, gen/dat/prep, ins), acc = nom.
const TEENS = {
  5: S("пять пяти пятью"), 6: S("шесть шести шестью"), 7: S("семь семи семью"),
  8: S("восемь восьми восемью"), 9: S("девять девяти девятью"), 10: S("десять десяти десятью"),
  11: S("одиннадцать одиннадцати одиннадцатью"), 12: S("двенадцать двенадцати двенадцатью"),
  13: S("тринадцать тринадцати тринадцатью"), 14: S("четырнадцать четырнадцати четырнадцатью"),
  15: S("пятнадцать пятнадцати пятнадцатью"), 16: S("шестнадцать шестнадцати шестнадцатью"),
  17: S("семнадцать семнадцати семнадцатью"), 18: S("восемнадцать восемнадцати восемнадцатью"),
  19: S("девятнадцать девятнадцати девятнадцатью"), 20: S("двадцать двадцати двадцатью"),
  30: S("тридцать тридцати тридцатью"),
  40: S("сорок сорока сорока"), 50: S("пятьдесят пятидесяти пятьюдесятью"),
  60: S("шестьдесят шестидесяти шестьюдесятью"), 70: S("семьдесят семидесяти семьюдесятью"),
  80: S("восемьдесят восьмидесяти восемьюдесятью"), 90: S("девяносто девяноста девяноста"),
};
const HUNDREDS = {
  100: S("сто ста ста сто ста ста"), 200: S("двести двухсот двумстам двести двумястами двухстах"),
  300: S("триста трёхсот трёмстам триста тремястами трёхстах"),
  400: S("четыреста четырёхсот четырёмстам четыреста четырьмястами четырёхстах"),
  500: S("пятьсот пятисот пятистам пятьсот пятьюстами пятистах"),
  600: S("шестьсот шестисот шестистам шестьсот шестьюстами шестистах"),
  700: S("семьсот семисот семистам семьсот семьюстами семистах"),
  800: S("восемьсот восьмисот восьмистам восемьсот восемьюстами восьмистах"),
  900: S("девятьсот девятисот девятистам девятьсот девятьюстами девятистах"),
};
const SCALES = [
  [1e9, S("миллиард миллиарда миллиарду миллиард миллиардом миллиарде"),
        S("миллиарды миллиардов миллиардам миллиарды миллиардами миллиардах"), M],
  [1e6, S("миллион миллиона миллиону миллион миллионом миллионе"),
        S("миллионы миллионов миллионам миллионы миллионами миллионах"), M],
  [1000, S("тысяча тысячи тысяче тысячу тысячей тысяче"),
         S("тысячи тысяч тысячам тысячи тысячами тысячах"), F],
];

const OBLIQUE = (i, c) => (c === NOM || c === ACC ? 0 : c === INS ? 2 : 1);

function small(n, c, g) {
  if (n === 0) return [ZERO[cidx(c)]];
  if (n === 1) return [ONE[g][cidx(c)]];
  if (n === 2) return [TWO[g][cidx(c)]];
  if (SMALL[n]) return [SMALL[n][cidx(c)]];
  const t = TEENS[n];
  if (t) return [t[OBLIQUE(n, c)]];
  if (n > 20) {
    const tens = Math.floor(n / 10) * 10;
    if (TEENS[tens]) return [TEENS[tens][OBLIQUE(tens, c)], ...small(n % 10, c, g)];
  }
  return [];
}

function underThousand(n, c, g) {
  if (n < 100) return small(n, c, g);
  const out = [HUNDREDS[Math.floor(n / 100) * 100][cidx(c)]];
  const rest = n % 100;
  if (rest) out.push(...small(rest, c, g));
  return out;
}

function scaleForm(count, singular, plural, c) {
  const mod100 = count % 100, mod10 = count % 10;
  if (mod10 === 1 && mod100 !== 11) return singular[cidx(c)];
  if (c !== NOM && c !== ACC) return plural[cidx(c)];
  if ([2, 3, 4].includes(mod10) && !(mod100 >= 11 && mod100 <= 14)) return singular[cidx(GEN)];
  return plural[cidx(GEN)];
}

export function cardinal(n, c = NOM, g = M) {
  if (n < 0) return ["минус", ...cardinal(-n, c, g)];
  if (n < 1000) return underThousand(n, c, g);
  const out = [];
  let rest = n;
  for (const [value, singular, plural, sg] of SCALES) {
    const count = Math.floor(rest / value);
    rest %= value;
    if (!count) continue;
    if (count === 1) out.push(singular[cidx(c)]);
    else out.push(...cardinal(count, c, sg), scaleForm(count, singular, plural, c));
  }
  if (rest) out.push(...underThousand(rest, c, g));
  return out;
}

export function agree(n, one, few, many) {
  const mod100 = n % 100, mod10 = n % 10;
  if (mod100 >= 11 && mod100 <= 14) return many;
  if (mod10 === 1) return one;
  if ([2, 3, 4].includes(mod10)) return few;
  return many;
}

// --- ordinals -------------------------------------------------------------

const ORD_STEMS = {
  0: "нулев", 1: "перв", 2: "втор", 3: "трет", 4: "четвёрт", 5: "пят", 6: "шест",
  7: "седьм", 8: "восьм", 9: "девят", 10: "десят", 11: "одиннадцат", 12: "двенадцат",
  13: "тринадцат", 14: "четырнадцат", 15: "пятнадцат", 16: "шестнадцат", 17: "семнадцат",
  18: "восемнадцат", 19: "девятнадцат", 20: "двадцат", 30: "тридцат", 40: "сороков",
  50: "пятидесят", 60: "шестидесят", 70: "семидесят", 80: "восьмидесят", 90: "девяност",
  100: "сот", 200: "двухсот", 300: "трёхсот", 400: "четырёхсот", 500: "пятисот",
  600: "шестисот", 700: "семисот", 800: "восьмисот", 900: "девятисот", 1000: "тысячн",
  2000: "двухтысячн", 3000: "трёхтысячн", 4000: "четырёхтысячн", 5000: "пятитысячн",
  1e6: "миллионн", 1e9: "миллиардн",
};
const SOFT_ENDINGS = new Set(["нулев", "втор", "шест", "седьм", "восьм", "сороков"]);
const END_M = S("ый ого ому ый ым ом"), END_F = S("ая ой ой ую ой ой"), END_N = S("ое ого ому ое ым ом");
const THIRD = {
  [`${M}${NOM}`]: "третий", [`${M}${GEN}`]: "третьего", [`${M}${DAT}`]: "третьему",
  [`${M}${ACC}`]: "третий", [`${M}${INS}`]: "третьим", [`${M}${PREP}`]: "третьем",
  [`${F}${NOM}`]: "третья", [`${F}${GEN}`]: "третьей", [`${F}${DAT}`]: "третьей",
  [`${F}${ACC}`]: "третью", [`${F}${INS}`]: "третьей", [`${F}${PREP}`]: "третьей",
  [`${N}${NOM}`]: "третье", [`${N}${GEN}`]: "третьего", [`${N}${DAT}`]: "третьему",
  [`${N}${ACC}`]: "третье", [`${N}${INS}`]: "третьим", [`${N}${PREP}`]: "третьем",
};

function ordWord(stem, c, g) {
  if (SOFT_ENDINGS.has(stem)) {
    const soft = { [`${M}${NOM}`]: "ой", [`${M}${ACC}`]: "ой" };
    const endings = g === F ? END_F : g === N ? END_N : END_M;
    return stem + (soft[`${g}${c}`] || endings[cidx(c)]);
  }
  if (stem === "трет") return THIRD[`${g}${c}`];
  return stem + (g === F ? END_F : g === N ? END_N : END_M)[cidx(c)];
}

export function ordinal(n, c = NOM, g = M) {
  if (n <= 0) return ordWord("нулев", c, g);
  if (ORD_STEMS[n]) return ordWord(ORD_STEMS[n], c, g);
  if (n < 100) {
    const tens = Math.floor(n / 10), units = n % 10;
    if (!units) return ordWord(ORD_STEMS[tens * 10], c, g);
    return cardinal(tens * 10, NOM, M).join(" ") + " " + ordWord(ORD_STEMS[units], c, g);
  }
  if (n < 1000) {
    const hundreds = Math.floor(n / 100), rest = n % 100;
    if (!rest) return ordWord(ORD_STEMS[hundreds * 100], c, g);
    return cardinal(hundreds * 100, NOM, M).join(" ") + " " + ordinal(rest, c, g);
  }
  for (const value of [1e9, 1e6, 1000]) {
    if (n < value) continue;
    const count = Math.floor(n / value), rest = n % value;
    if (!rest) {
      if (ORD_STEMS[n]) return ordWord(ORD_STEMS[n], c, g);
      if (count === 1) return ordWord(ORD_STEMS[value], c, g);
      return cardinal(n, NOM, M).join(" ") + " " + ordWord(ORD_STEMS[value], c, g);
    }
    return cardinal(n - rest, NOM, M).join(" ") + " " + ordinal(rest, c, g);
  }
  return cardinal(n, c, g).join(" ");
}

// --- decimals -------------------------------------------------------------

const FRACTION_ONE = { [M]: "целый", [F]: "целая", [N]: "целое" };
const DEN = {
  10: ["десятая", "десятых", "десятых"],
  100: ["сотая", "сотых", "сотых"],
  1000: ["тысячная", "тысячных", "тысячных"],
};
const form = (value, [one, few, many]) =>
  value % 10 === 1 && value % 100 !== 11 ? one
    : [2, 3, 4].includes(value % 10) && !(value % 100 >= 11 && value % 100 <= 14) ? few : many;

export function decimal(value, c = NOM, g = M) {
  const [whole, rawFraction = ""] = value.split(",");
  const integer = parseInt(whole, 10);
  const fraction = rawFraction.replace(/0+$/, "");
  if (!fraction) return cardinal(integer, c, g);
  if (integer === 1 && fraction === "5") return [g === F ? "полторы" : "полтора"];
  const out = cardinal(integer, c, g);
  const fracValue = parseInt(fraction, 10);
  const den = DEN[10 ** fraction.length] || DEN[10];
  if (integer !== 0) out.push(integer % 10 === 1 && integer % 100 !== 11 ? FRACTION_ONE[g] : form(integer, ["", "целых", "целых"]));
  out.push(...cardinal(fracValue, NOM, F), form(fracValue, den));
  return out;
}

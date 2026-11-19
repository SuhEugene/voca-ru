"""Russian cardinal and ordinal numerals with case and gender agreement.

Scope: the cases the normaliser can infer from context (`nom`, `gen`, `dat`,
`acc`, `ins`, `prep`). Accusative uses the nominative form for inanimate
number words and the genitive form for 1–4 masculine (the animate reading);
animacy beyond that is not inferred.
"""

NOM, GEN, DAT, ACC, INS, PREP = "nom", "gen", "dat", "acc", "ins", "prep"
CASES = (NOM, GEN, DAT, ACC, INS, PREP)
M, F, N = "m", "f", "n"

ZERO = ("ноль", "ноля", "нолю", "ноль", "нолём", "ноле")

# gender -> (nom, gen, dat, acc, ins, prep)
ONE = {
    M: ("один", "одного", "одному", "один", "одним", "одном"),
    F: ("одна", "одной", "одной", "одну", "одной", "одной"),
    N: ("одно", "одного", "одному", "одно", "одним", "одном"),
}
TWO = {
    M: ("два", "двух", "двум", "два", "двумя", "двух"),
    F: ("две", "двух", "двум", "две", "двумя", "двух"),
    N: ("два", "двух", "двум", "два", "двумя", "двух"),
}
SMALL = {
    3: ("три", "трёх", "трём", "три", "тремя", "трёх"),
    4: ("четыре", "четырёх", "четырём", "четыре", "четырьмя", "четырёх"),
}

# 5–20, 30: (nom, gen/dat/prep, ins), acc = nom.
TEENS = {
    5: ("пять", "пяти", "пятью"), 6: ("шесть", "шести", "шестью"),
    7: ("семь", "семи", "семью"), 8: ("восемь", "восьми", "восемью"),
    9: ("девять", "девяти", "девятью"), 10: ("десять", "десяти", "десятью"),
    11: ("одиннадцать", "одиннадцати", "одиннадцатью"),
    12: ("двенадцать", "двенадцати", "двенадцатью"),
    13: ("тринадцать", "тринадцати", "тринадцатью"),
    14: ("четырнадцать", "четырнадцати", "четырнадцатью"),
    15: ("пятнадцать", "пятнадцати", "пятнадцатью"),
    16: ("шестнадцать", "шестнадцати", "шестнадцатью"),
    17: ("семнадцать", "семнадцати", "семнадцатью"),
    18: ("восемнадцать", "восемнадцати", "восемнадцатью"),
    19: ("девятнадцать", "девятнадцати", "девятнадцатью"),
    20: ("двадцать", "двадцати", "двадцатью"),
    30: ("тридцать", "тридцати", "тридцатью"),
}
TENS_ODD = {
    40: ("сорок", "сорока", "сорока"),
    50: ("пятьдесят", "пятидесяти", "пятьюдесятью"),
    60: ("шестьдесят", "шестидесяти", "шестьюдесятью"),
    70: ("семьдесят", "семидесяти", "семьюдесятью"),
    80: ("восемьдесят", "восьмидесяти", "восемьюдесятью"),
    90: ("девяносто", "девяноста", "девяноста"),
}
HUNDREDS = {
    100: ("сто", "ста", "ста", "сто", "ста", "ста"),
    200: ("двести", "двухсот", "двумстам", "двести", "двумястами", "двухстах"),
    300: ("триста", "трёхсот", "трёмстам", "триста", "тремястами", "трёхстах"),
    400: ("четыреста", "четырёхсот", "четырёмстам", "четыреста", "четырьмястами", "четырёхстах"),
    500: ("пятьсот", "пятисот", "пятистам", "пятьсот", "пятьюстами", "пятистах"),
    600: ("шестьсот", "шестисот", "шестистам", "шестьсот", "шестьюстами", "шестистах"),
    700: ("семьсот", "семисот", "семистам", "семьсот", "семьюстами", "семистах"),
    800: ("восемьсот", "восьмисот", "восьмистам", "восемьсот", "восемьюстами", "восьмистах"),
    900: ("девятьсот", "девятисот", "девятистам", "девятьсот", "девятьюстами", "девятистах"),
}

CIDX = {c: i for i, c in enumerate(CASES)}

# scale: (value, singular forms, plural forms, gender)
SCALES = [
    (10 ** 9, ("миллиард", "миллиарда", "миллиарду", "миллиард", "миллиардом", "миллиарде"),
     ("миллиарды", "миллиардов", "миллиардам", "миллиарды", "миллиардами", "миллиардах"), M),
    (10 ** 6, ("миллион", "миллиона", "миллиону", "миллион", "миллионом", "миллионе"),
     ("миллионы", "миллионов", "миллионам", "миллионы", "миллионами", "миллионах"), M),
    (1000, ("тысяча", "тысячи", "тысяче", "тысячу", "тысячей", "тысяче"),
     ("тысячи", "тысяч", "тысячам", "тысячи", "тысячами", "тысячах"), F),
]

COLLECTIVE = {2: "двое", 3: "трое", 4: "четверо", 5: "пятеро",
              6: "шестеро", 7: "семеро", 8: "восьмеро", 9: "девятеро",
              10: "десятеро"}


def _small(n, case, gender):
    if n == 0:
        return [ZERO[CIDX[case]]]
    if n == 1:
        return [ONE[gender][CIDX[case]]]
    if n == 2:
        return [TWO[gender][CIDX[case]]]
    if n in SMALL:
        return [SMALL[n][CIDX[case]]]
    if n in TEENS:
        nom, oblique, ins = TEENS[n]
        return [nom if case in (NOM, ACC) else ins if case == INS else oblique]
    if n in TENS_ODD:
        nom, oblique, ins = TENS_ODD[n]
        return [nom if case in (NOM, ACC) else ins if case == INS else oblique]
    if n > 20:
        tens, units = divmod(n, 10)
        tens *= 10
        if tens in TEENS:
            nom, oblique, ins = TEENS[tens]
        else:
            nom, oblique, ins = TENS_ODD[tens]
        out = [nom if case in (NOM, ACC) else ins if case == INS else oblique]
        out.extend(_small(units, case, gender))
        return out
    return []


def _hundreds(n, case):
    if n in HUNDREDS:
        return [HUNDREDS[n][CIDX[case]]]
    hundreds, rest = divmod(n, 100)
    out = []
    if hundreds:
        out.append(HUNDREDS[hundreds * 100][CIDX[case]])
    out.extend(_small(rest, case, M))
    return out


def _under_thousand(n, case, gender=M):
    """0..999 -> words, case applied."""
    if n < 100:
        return _small(n, case, gender)
    hundreds, rest = divmod(n, 100)
    out = [HUNDREDS[hundreds * 100][CIDX[case]]]
    if rest:
        out.extend(_small(rest, case, gender))
    return out


def _scale_form(count, singular, plural, case):
    """Which scale form follows ``count`` in ``case``.

    Nominative/accusative: 1 -> singular, 2-4 -> genitive singular,
    5+ -> genitive plural. Oblique cases: 1 -> singular, else plural in that
    case ("до двух тысяч", "в пяти тысячах").
    """
    mod100 = count % 100
    mod10 = count % 10
    oblique = case not in (NOM, ACC)
    if mod10 == 1 and mod100 != 11:
        return singular[CIDX[case]]
    if oblique:
        return plural[CIDX[case]]
    if mod10 in (2, 3, 4) and mod100 not in range(11, 15):
        return singular[CIDX[GEN]]
    return plural[CIDX[GEN]]


def cardinal(n, case=NOM, gender=M):
    """Integer -> Russian words. ``case`` applies to every word."""
    if n < 0:
        return ["минус"] + cardinal(-n, case, gender)
    if n < 1000:
        return _under_thousand(n, case, gender)
    out = []
    rest = n
    for value, singular, plural, scale_gender in SCALES:
        count, rest = divmod(rest, value)
        if count:
            if count == 1:
                out.append(singular[CIDX[case]])
            else:
                out.append("")
                out[-1:] = cardinal(count, case, scale_gender)
                out.append(_scale_form(count, singular, plural, case))
    if rest:
        out.extend(_under_thousand(rest, case, gender))
    return out


def agree(n, one, few, many):
    """Pick the noun form that follows cardinal ``n``."""
    mod100 = n % 100
    mod10 = n % 10
    if mod100 in range(11, 15):
        return many
    if mod10 == 1:
        return one
    if mod10 in (2, 3, 4):
        return few
    return many


# --- ordinals -------------------------------------------------------------

# Stem + hard/soft adjective endings.
_ORD_STEMS = {
    0: "нулев", 1: "перв", 2: "втор", 3: "трет", 4: "четвёрт",
    5: "пят", 6: "шест", 7: "седьм", 8: "восьм", 9: "девят",
    10: "десят", 11: "одиннадцат", 12: "двенадцат", 13: "тринадцат",
    14: "четырнадцат", 15: "пятнадцат", 16: "шестнадцат", 17: "семнадцат",
    18: "восемнадцат", 19: "девятнадцат", 20: "двадцат", 30: "тридцат",
    40: "сороков", 50: "пятидесят", 60: "шестидесят", 70: "семидесят",
    80: "восьмидесят", 90: "девяност", 100: "сот", 200: "двухсот",
    300: "трёхсот", 400: "четырёхсот", 500: "пятисот", 600: "шестисот",
    700: "семисот", 800: "восьмисот", 900: "девятисот", 1000: "тысячн",
    2000: "двухтысячн", 3000: "трёхтысячн", 4000: "четырёхтысячн",
    5000: "пятитысячн", 1000000: "миллионн", 1000000000: "миллиардн",
}
_SOFT_ENDINGS = {  # -ой stems: второй, шестой, седьмой, восьмой, нулевой ...
    "нулев", "втор", "шест", "седьм", "восьм", "сороков",
}
_MASC = ("ый", "ого", "ому", "ый", "ым", "ом")
_FEM = ("ая", "ой", "ой", "ую", "ой", "ой")
_NEUT = ("ое", "ого", "ому", "ое", "ым", "ом")
_MASC_SOFT = ("ой", "ого", "ому", "ой", "ым", "ом")
_FEM_SOFT = ("ая", "ой", "ой", "ую", "ой", "ой")
_NEUT_SOFT = ("ое", "ого", "ому", "ое", "ым", "ом")


def _ord_word(stem, case, gender):
    key = (gender, case)
    if stem in _SOFT_ENDINGS:
        table = {("m", NOM): "ой", ("m", GEN): "ого", ("m", DAT): "ому",
                 ("m", ACC): "ой", ("m", INS): "ым", ("m", PREP): "ом",
                 ("f", NOM): "ая", ("f", GEN): "ой", ("f", DAT): "ой",
                 ("f", ACC): "ую", ("f", INS): "ой", ("f", PREP): "ой",
                 ("n", NOM): "ое", ("n", GEN): "ого", ("n", DAT): "ому",
                 ("n", ACC): "ое", ("n", INS): "ым", ("n", PREP): "ом"}
        return stem + table[key]
    if stem == "трет":
        table = {("m", NOM): "третий", ("m", GEN): "третьего",
                 ("m", DAT): "третьему", ("m", ACC): "третий",
                 ("m", INS): "третьим", ("m", PREP): "третьем",
                 ("f", NOM): "третья", ("f", GEN): "третьей",
                 ("f", DAT): "третьей", ("f", ACC): "третью",
                 ("f", INS): "третьей", ("f", PREP): "третьей",
                 ("n", NOM): "третье", ("n", GEN): "третьего",
                 ("n", DAT): "третьему", ("n", ACC): "третье", ("n", INS): "третьим",
                 ("n", PREP): "третьем"}
        return table[key]
    endings = {M: _MASC, F: _FEM, N: _NEUT}[gender]
    return stem + endings[CIDX[case]]


def ordinal(n, case=NOM, gender=M):
    """Integer -> one ordinal word; only the last component is ordinalised."""
    if n <= 0:
        return _ord_word("нулев", case, gender)
    if n in _ORD_STEMS:
        return _ord_word(_ORD_STEMS[n], case, gender)
    if 11 <= n <= 19:
        return _ord_word(_ORD_STEMS[n], case, gender)
    if n < 100:
        tens, units = divmod(n, 10)
        if units == 0:
            return _ord_word(_ORD_STEMS[tens * 10], case, gender)
        return " ".join(cardinal(tens * 10, NOM, M)) + " " + _ord_word(
            _ORD_STEMS[units], case, gender)
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if rest == 0:
            return _ord_word(_ORD_STEMS[hundreds * 100], case, gender)
        return " ".join(cardinal(hundreds * 100, NOM, M)) + " " + ordinal(rest, case, gender)
    for value, _name in ((10 ** 9, "миллиард"), (10 ** 6, "миллион"), (1000, "тысяч")):
        if n >= value:
            count, rest = divmod(n, value)
            if rest == 0:
                if n in _ORD_STEMS:
                    return _ord_word(_ORD_STEMS[n], case, gender)
                if count == 1:
                    return _ord_word(_ORD_STEMS[value], case, gender)
                return " ".join(cardinal(n, NOM, M)) + " " + _ord_word(
                    _ORD_STEMS[value], case, gender)
            head = " ".join(cardinal(n - rest, NOM, M))
            return head + " " + ordinal(rest, case, gender)
    return " ".join(cardinal(n, case, gender))


# --- decimals -------------------------------------------------------------

FRACTION_ONE = {
    M: "целый", F: "целая", N: "целое",
}
FRACTION_FEW = "целых"
FRACTION_MANY = "целых"


def decimal(value, case=NOM, gender=M):
    """'2,5' -> две целых пять десятых (as words)."""
    integer, fraction = value.split(",", 1)
    integer = int(integer)
    fraction = fraction.rstrip("0")
    if not fraction:
        return cardinal(integer, case, gender)
    if integer == 1 and fraction == "5":
        # Noun keeps whatever genitive-singular form the source text has.
        return ["полтора" if gender in (M, N) else "полторы"]
    whole = cardinal(integer, case, gender)
    frac_value = int(fraction)
    frac_den = 10 ** len(fraction)
    den_word = {
        10: {1: "десятая", 2: "десятых", 5: "десятых"},
        100: {1: "сотая", 2: "сотых", 5: "сотых"},
        1000: {1: "тысячная", 2: "тысячных", 5: "тысячных"},
    }.get(frac_den)
    if den_word is None:
        den_word = {1: "десятая", 2: "десятых", 5: "десятых"}
    if integer == 0:
        whole_word = []
    else:
        whole_word = [FRACTION_ONE[gender] if integer % 10 == 1 and integer % 100 != 11
                      else FRACTION_FEW if integer % 10 in (2, 3, 4)
                      and integer % 100 not in range(11, 15) else FRACTION_MANY]
    frac = cardinal(frac_value, NOM, F)
    frac.append(den_word[1] if frac_value % 10 == 1 and frac_value % 100 != 11
                else den_word[2] if frac_value % 10 in (2, 3, 4)
                and frac_value % 100 not in range(11, 15) else den_word[5])
    return whole + whole_word + frac

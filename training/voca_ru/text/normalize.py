"""Deterministic Russian text normalisation for the phoneme path.

Turns raw UTF-8 text into lowercase words and inventory punctuation:
numbers, dates, times, money, ordinals, abbreviations and Latin runs are
expanded; ``ё`` is preserved; no digits or Latin letters remain.

The same rules are what the eval harness must use (contract §5), so this
module has no dependency outside the standard library and is idempotent.
"""

import re
import unicodedata

from . import numbers

MONTHS_GEN = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]
MONTH_RE = "(?:" + "|".join(MONTHS_GEN) + ")"

VOWELS = "аеёиоуыэюя"

LATIN_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф",
    "g": "джи", "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл",
    "m": "эм", "n": "эн", "o": "о", "p": "пи", "q": "кью", "r": "ар",
    "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл ю", "x": "экс",
    "y": "уай", "z": "зет",
}

# unit -> (one, few, many)
UNITS = {
    "км": ("километр", "километра", "километров"),
    "м": ("метр", "метра", "метров"),
    "см": ("сантиметр", "сантиметра", "сантиметров"),
    "мм": ("миллиметр", "миллиметра", "миллиметров"),
    "кг": ("килограмм", "килограмма", "килограммов"),
    "г": ("грамм", "грамма", "граммов"),
    "мг": ("миллиграмм", "миллиграмма", "миллиграммов"),
    "л": ("литр", "литра", "литров"),
    "мл": ("миллилитр", "миллилитра", "миллилитров"),
    "га": ("гектар", "гектара", "гектаров"),
    "руб": ("рубль", "рубля", "рублей"),
    "коп": ("копейка", "копейки", "копеек"),
}

DIGIT_WORDS = ["ноль", "один", "два", "три", "четыре",
               "пять", "шесть", "семь", "восемь", "девять"]

GEN_PREPS = {"до", "от", "из", "без", "около", "для", "у", "после",
             "кроме", "ради", "больше", "меньше", "свыше", "порядка"}
DAT_PREPS = {"к", "по"}
PREP_PREPS = {"в", "на", "о", "об", "при"}
INS_PREPS = {"с", "между", "над", "под", "перед"}


def _clean(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    # "+" before a vowel is how a user marks stress by hand; it becomes the acute
    text = re.sub(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])", lambda m: m.group(1) + "\u0301", text)
    text = text.lower().replace("ё", "ё")
    for bad in "\u200b\u200c\u200d\ufeff\t\r\n":
        text = text.replace(bad, " ")
    return text


def _merge_thousands(text):
    """'1 500 000' -> '1500000' before number parsing."""
    return re.sub(r"(?<!\d)(\d{1,3})(?:[ ]\d{3})+(?!\d)",
                  lambda m: m.group(0).replace(" ", ""), text)


SCALE_UNITS = {
    "тыс.": ("тысяча", "тысячи", "тысяч", numbers.F),
    "тыс": ("тысяча", "тысячи", "тысяч", numbers.F),
    "млн": ("миллион", "миллиона", "миллионов", numbers.M),
    "млрд": ("миллиард", "миллиарда", "миллиардов", numbers.M),
}


def _expand_scales(text):
    pattern = re.compile(r"(?<!\d)(\d{1,9})\s*(тыс\.|тыс\b|млн\b|млрд\b)")

    def repl(match):
        value, unit = int(match.group(1)), match.group(2)
        one, few, many, gender = SCALE_UNITS[unit]
        return " ".join(numbers.cardinal(value, numbers.NOM, gender) + [
            numbers.agree(value, one, few, many)])

    return pattern.sub(repl, text)


def _expand_abbreviations(text):
    rules = [
        (r"т\.\s*д\.", "так далее"),
        (r"т\.\s*е\.", "то есть"),
        (r"т\.\s*к\.", "так как"),
        (r"т\.\s*п\.", "тому подобное"),
        (r"и\s+т\.\s*д\.", "и так далее"),
        (r"гг\.", "годы"),
        (r"см\.", "смотри"),
        (r"стр\.", "страница"),
        (r"ул\.", "улица"),
        (r"пр\.", "проспект"),
        (r"кв\.", "квартира"),
        (r"им\.", "имени"),
        (r"проф\.", "профессор"),
        (r"тов\.", "товарищ"),
        (r"тел\.", "телефон"),
        (r"рис\.", "рисунок"),
        (r"табл\.", "таблица"),
        (r"тыс\.", "тысяч"),
        (r"млн", "миллион"),
        (r"млрд", "миллиард"),
        (r"руб\.", "рублей"),
        (r"коп\.", "копеек"),
        (r"др\.", "другое"),
        (r"№", "номер"),
    ]
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text)
    # Contextual sense first: г. after a year is "года", otherwise "город";
    # с. and д. before a number are "страница"/"дом".
    text = re.sub(r"(?<=\d)\s+г\.", " года", text)
    text = re.sub(r"(?<=\d)\s+с\.", " страница", text)
    text = re.sub(r"(?<=\d)\s+д\.", " дом", text)
    text = re.sub(r"\bг\.", "город", text)
    text = re.sub(r"\bс\.", "страница", text)
    text = re.sub(r"\bд\.", "дом", text)
    return text


MASCULINE_GENITIVES = {
    "часа", "дня", "года", "месяца", "рубля", "метра", "километра",
    "литра", "процента", "килограмма", "грамма", "доллара", "миллиона",
}


def _gender_of(word):
    word = word.strip(".,!?;:«»\"'()[]{}—–…")
    if not word:
        return numbers.M
    if word in MASCULINE_GENITIVES:
        return numbers.M
    if word.endswith(("ая", "яя")):
        return numbers.F
    if word.endswith(("ое", "ее", "мя")):
        return numbers.N
    if word.endswith(("а", "я", "ы", "и")):
        return numbers.F
    if word.endswith(("о", "е")):
        return numbers.N
    return numbers.M


def _nearest_word_before(text, pos):
    head = text[:pos].rstrip()
    if not head:
        return ""
    return head.split()[-1].strip(".,!?;:«»\"'()[]{}—–…").lower()


def _word_after(text, pos):
    rest = text[pos:].lstrip()
    match = re.match(r"[а-яё-]+", rest)
    return match.group(0) if match else ""


def _case_for(text, pos, lookahead=True, after=None):
    prep = _nearest_word_before(text, pos)
    if prep in GEN_PREPS:
        return numbers.GEN
    if prep in DAT_PREPS:
        return numbers.DAT
    if prep in INS_PREPS:
        return numbers.INS
    if prep in PREP_PREPS:
        nxt = _word_after(text, after if after is not None else pos) if lookahead else ""
        if prep in {"в", "на"} and not nxt.endswith(("ах", "ях", "е", "и")):
            return numbers.ACC
        return numbers.PREP
    return numbers.NOM


def _apply_case(text, pos, gender=numbers.M, lookahead=True):
    return numbers.cardinal(int(text), _case_for(text, pos, lookahead), gender)


def _day_ordinal(day):
    return numbers.ordinal(int(day), numbers.GEN, numbers.M)


def _year_ordinal(year, case):
    return numbers.ordinal(int(year), case, numbers.M)


def _expand_dates(text):
    def dot(match):
        day, month, year = match.group(1), int(match.group(2)), match.group(3)
        if not 1 <= month <= 12:
            return match.group(0)
        return f"{_day_ordinal(day)} {MONTHS_GEN[month - 1]} {_year_ordinal(year, numbers.GEN)} года"

    text = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", dot, text)

    def words(match):
        day, month, year = match.group(1), match.group(2), match.group(3)
        out = f"{_day_ordinal(day)} {month}"
        if year:
            out += f" {_year_ordinal(year, numbers.GEN)} года"
        return out

    return re.sub(rf"\b(\d{{1,2}})\s+({MONTH_RE})(?:\s+(\d{{4}})(?:\s+(?:года|году|год))?)?\b", words, text)


def _expand_time(text):
    def repl(match):
        hour, minute = int(match.group(1)), match.group(2)
        if hour > 23 or int(minute) > 59:
            return match.group(0)
        words = numbers.cardinal(hour, numbers.NOM, numbers.M)
        if minute == "00":
            words += ["ноль", "ноль"]
        elif minute.startswith("0"):
            words += ["ноль"] + numbers.cardinal(int(minute), numbers.NOM, numbers.M)
        else:
            words += numbers.cardinal(int(minute), numbers.NOM, numbers.M)
        return " ".join(words)

    return re.sub(r"\b(\d{1,2}):(\d{2})\b", repl, text)


def _expand_money(text):
    def decimal_money(match):
        whole, unit = int(match.group(1)), match.group(3)
        # "2.1$" is two dollars ten cents: a single decimal digit is tenths
        minor = int(match.group(2).ljust(2, "0"))
        if unit in ("$", "€"):
            words = numbers.cardinal(whole, numbers.NOM, numbers.M)
            if unit == "$":
                words.append(numbers.agree(whole, "доллар", "доллара", "долларов"))
            else:
                words.append("евро")
            if minor:
                words.extend(numbers.cardinal(minor, numbers.NOM, numbers.M))
                words.append(numbers.agree(minor, "цент", "цента", "центов"))
            return " ".join(words)
        case = _case_for(text, match.start(), lookahead=False)
        words = numbers.cardinal(whole, case, numbers.M)
        words.append(numbers.agree(whole, "рубль", "рубля", "рублей"))
        if minor:
            words.extend(numbers.cardinal(minor, numbers.NOM, numbers.F))
            words.append(numbers.agree(minor, "копейка", "копейки", "копеек"))
        return " ".join(words)

    # 2.1$ / 5,50€ / 169.99₽ — the fraction belongs to a named subunit
    text = re.sub(r"(?<!\d)(\d{1,9})[.,](\d{1,2})\s*([$€₽])", decimal_money, text)
    text = re.sub(r"(?<!\d)(\d{1,9})[.,](\d{1,2})\s*(руб(?:\.|ль|ля|лей)?)\b", decimal_money, text)

    def rubles(match):
        value = int(match.group(1))
        case = _case_for(text, match.start(), lookahead=False)
        words = numbers.cardinal(value, case, numbers.M)
        words.append(numbers.agree(value, "рубль", "рубля", "рублей"))
        return " ".join(words)

    text = re.sub(r"(?<!\d)(\d{1,9})\s*(?:₽|руб(?:\.|ль|ля|лей)?\b)", rubles, text)

    def kopecks(match):
        value = int(match.group(1))
        case = _case_for(text, match.start(), lookahead=False)
        words = numbers.cardinal(value, case, numbers.F)
        words.append(numbers.agree(value, "копейка", "копейки", "копеек"))
        return " ".join(words)

    text = re.sub(r"(?<!\d)(\d{1,9})\s*(?:коп\.?|копейка|копейки|копеек)\b", kopecks, text)
    text = re.sub(r"(?<!\d)(\d{1,9})\s*€", lambda m: " ".join(
        numbers.cardinal(int(m.group(1)), numbers.NOM, numbers.M) + ["евро"]), text)
    def dollars(match):
        value = int(match.group(1))
        return " ".join(numbers.cardinal(value, numbers.NOM, numbers.M) + [
            numbers.agree(value, "доллар", "доллара", "долларов")])

    text = re.sub(r"(?<!\d)(\d{1,9})\s*\$", dollars, text)
    return text


def _expand_percent(text):
    def repl(match):
        value = int(match.group(1))
        return " ".join(numbers.cardinal(value, numbers.NOM, numbers.M) + [
            numbers.agree(value, "процент", "процента", "процентов")])

    return re.sub(r"(?<!\d)(\d{1,9})\s*%", repl, text)


ORDINAL_SUFFIX = {
    "й": (numbers.NOM, numbers.M), "ый": (numbers.NOM, numbers.M),
    "ой": (numbers.NOM, numbers.M), "го": (numbers.GEN, numbers.M),
    "му": (numbers.DAT, numbers.M), "м": (numbers.INS, numbers.M),
    "х": (numbers.PREP, numbers.M), "я": (numbers.NOM, numbers.F),
    "ю": (numbers.ACC, numbers.F), "е": (numbers.NOM, numbers.N),
    "ая": (numbers.NOM, numbers.F), "ое": (numbers.NOM, numbers.N),
}


def _expand_ordinals(text):
    def repl(match):
        value = int(match.group(1))
        suffix = match.group(2)
        case, gender = ORDINAL_SUFFIX.get(suffix, (numbers.NOM, numbers.M))
        return numbers.ordinal(value, case, gender)

    return re.sub(r"(?<!\d)(\d{1,4})-?(й|ый|ой|го|му|м|х|я|ю|е|ая|ое)\b", repl, text)


def _expand_years(text):
    def repl(match):
        year, suffix = match.group(1), match.group(2)
        case = numbers.PREP if suffix == "году" else numbers.GEN if suffix == "года" else numbers.NOM
        return f"{_year_ordinal(year, case)} {suffix}"

    return re.sub(r"\b(\d{4})\s+(году|года|год)\b", repl, text)


def _expand_units(text):
    unit_re = "|".join(sorted(UNITS, key=len, reverse=True))
    pattern = re.compile(rf"(?<!\d)(\d{{1,9}})\s*({unit_re})\b")

    def repl(match):
        value, unit = int(match.group(1)), match.group(2)
        one, few, many = UNITS[unit]
        case = _case_for(text, match.start(), lookahead=False)
        return " ".join(numbers.cardinal(value, case, numbers.M) + [
            numbers.agree(value, one, few, many)])

    return pattern.sub(repl, text)


def _expand_decimals(text):
    def repl(match):
        raw = match.group(0)
        gender = _gender_of(_word_after(text, match.end()))
        case = _case_for(text, match.start(), lookahead=False)
        return " ".join(numbers.decimal(raw, case, gender))

    return re.sub(r"(?<!\d)(\d{1,3},\d{1,3})(?!\d)", repl, text)


def _digit_by_digit(value):
    return [DIGIT_WORDS[int(d)] for d in value if d.isdigit()]


def _expand_phones(text):
    """'по номеру 495 123 45 67' -> digit-by-digit, before thousands merge."""
    pattern = re.compile(r"(номер[а-яё]*|телефон[а-яё]*|тел)\s+(\d[\d\s-]*\d)")

    def repl(match):
        digits = "".join(char for char in match.group(2) if char.isdigit())
        return match.group(1) + " " + " ".join(_digit_by_digit(digits))

    return pattern.sub(repl, text)


def _expand_long_numbers(text):
    context = re.compile(r"(?:инн|огрн|снилс|номер|телефон|тел|счёт|счет|билет|платформа)\s*$")

    def repl(match):
        value = match.group(0)
        before = text[:match.start()]
        if len(value) >= 8 or context.search(before[-40:].lower()):
            return " ".join(_digit_by_digit(value))
        case = _case_for(text, match.start(), after=match.end())
        gender = _gender_of(_word_after(text, match.end()))
        return " ".join(numbers.cardinal(int(value), case, gender))

    return re.sub(r"(?<!\d)\d+(?!\d)", repl, text)


def _expand_latin(text):
    def repl(match):
        return " ".join(LATIN_NAMES.get(char, char) for char in match.group(0))

    return re.sub(r"[a-z]+", repl, text)


def _punctuation(text):
    text = re.sub(r"\.{2,}", "…", text)
    text = text.replace("„", "«").replace("“", "«").replace("”", "»")
    text = text.replace("–", "—").replace("−", "-").replace("’", "'").replace("`", "'")
    # Alternate ASCII double quotes so they become a matched pair.
    state = {"open": True}

    def quote(_match):
        state["open"] = not state["open"]
        return "»" if state["open"] else "«"

    text = re.sub(r'"', quote, text)
    allowed = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя\u0301 ,.!?;:\"'()—…«»-–№%[]{}")
    text = "".join(char if char in allowed else " " for char in text)
    return text


def _spaces(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?…»)\]])", r"\1", text)
    text = re.sub(r"([«(\[])\s+", r"\1", text)
    return text.strip()


def normalize(text):
    """Raw text -> lowercase words and inventory punctuation."""
    text = _clean(text)
    text = _expand_phones(text)
    text = _merge_thousands(text)
    text = _expand_scales(text)
    text = _expand_abbreviations(text)
    text = _expand_dates(text)
    text = _expand_time(text)
    text = _expand_years(text)
    text = _expand_money(text)
    text = _expand_percent(text)
    text = _expand_ordinals(text)
    text = _expand_units(text)
    text = _expand_decimals(text)
    text = _expand_long_numbers(text)
    text = _expand_latin(text)
    text = _punctuation(text)
    return _spaces(text)

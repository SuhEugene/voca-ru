#!/usr/bin/env python3
"""Build the frozen Russian script corpus (F2).

Regenerates data/scripts-ru/batch1.jsonl, batch2.jsonl and SPLITS.sha256 from
recorded sources. The build is seeded and byte-identical when the pinned source
files are present in the cache (the pinned SHA-256 values below are checked).

Sources:
  * Common Voice ru sentence-collector.txt (CC0-1.0)
  * ru.wikisource.org pages at pinned revisions (public-domain prose)
  * scripts/text/own/batch{1,2}.txt (project-own, CC0 waiver)
  * sentence templates in this file (project-own, CC0 waiver)

Cache: $VOCA_RU_CACHE (default: the project .cache directory, see AGENTS.md).
Nothing is written outside data/scripts-ru/ and the cache.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "scripts-ru"
OWN_DIR = ROOT / "scripts" / "text" / "own"
CACHE = Path(os.environ.get("VOCA_RU_CACHE", "/Users/suheugene/projects/22-tiniest-voca-ru/.cache"))
TEXT_CACHE = CACHE / "text"
WS_CACHE = TEXT_CACHE / "wikisource"

SEED = 20261119
SPLIT_SEED = 20261120
VAL_PER_1000 = 70  # -> 315/315 of 4500

UA = "voca-ru-corpus-builder/1.0 (https://github.com/SuhEugene/voca-ru; research dataset)"

# ---------------------------------------------------------------------------
# Pinned sources
# ---------------------------------------------------------------------------

CV_URL = "https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/ru/sentence-collector.txt"
CV_SHA256 = "e0869a008310c452f63fd02e4d52f8abc58df2de73e558c486719161e3ad1b87"
CV_LICENSE = "cc0-1.0"

# (slug, title, author, died, written, revid, sha256 of the cached parse JSON)
WIKISOURCE_PAGES = [
    ("pikovaia-dama", "Пиковая дама (Пушкин)", "А. С. Пушкин", 1837, 1834, 5644345,
     "0e972157e8362a2680b9ba34dc743550678cb2e4b610f1a624ade908d951dc3c"),
    ("asia", "Ася (Тургенев)", "И. С. Тургенев", 1883, 1858, 5759963,
     "3e1303cd1ee6f5d649297053cf70175cf816903b404a5914b41fc15141942984"),
    ("belye-nochi", "Белые ночи (Достоевский)", "Ф. М. Достоевский", 1881, 1848, 5733096,
     "cb10844bad5a635aa79ffe224978a4680aefa2a34f6b96f1ed8eec87e0ca9ca1"),
    ("mumu", "Муму (Тургенев)", "И. С. Тургенев", 1883, 1854, 5619020,
     "ff2038b469238360b2ac8952aab0fbf434a56cf3fc3fda598c5d4d7755e9a67c"),
    ("ionych", "Ионыч (Чехов)", "А. П. Чехов", 1904, 1898, 5321162,
     "8c788a0e6fb62068e3cce0ed7c17c1c213c477c2163d574ddca887993e4023d0"),
    ("dama-s-sobachkoi", "Дама с собачкой (Чехов)", "А. П. Чехов", 1904, 1899, 5736015,
     "f4e74befd252d908057b963e8fe06b1adc470cb7e475ca2b7ea3fd9666d12113"),
    ("posle-bala", "После бала (Толстой)", "Л. Н. Толстой", 1910, 1911, 5638188,
     "be2c3af162b6517ed2ac7f529cc64e7fb44aa6abb23f09e6767113ac24c32f17"),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Text filtering
# ---------------------------------------------------------------------------

# Russian letters, space and ordinary punctuation. No digits, no Latin.
ALLOWED = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
              "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
              " .,;:!?«»„“”\"()—–-")
BAD_SUBSTRINGS = ("http", "www", "..", ",,", "??", "!!", ";;", "()")


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u2009", " ").replace("\u202f", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[—–-]\s*", "", text)
    text = re.sub(r"^[«»\"“”„]\s*", "", text)
    text = re.sub(r"\s*[«»\"“”„]$", "", text)
    return text.strip()


def text_problem(text: str) -> str | None:
    """Return a rejection reason, or None when the sentence is usable."""
    if not text:
        return "empty"
    words = text.split()
    if not 5 <= len(words) <= 30:
        return "word count"
    if text[-1] not in ".!?":
        return "no final punctuation"
    if text.count("«") != text.count("»"):
        return "unbalanced quotes"
    for ch in text:
        if ch not in ALLOWED:
            return f"char {ch!r}"
    low = text.lower()
    for bad in BAD_SUBSTRINGS:
        if bad in low:
            return f"repeated punct {bad!r}"
    if not text[0].isupper():
        return "lowercase start"
    return None


def norm_key(text: str) -> str:
    """Comparison key: case-, ё- and punctuation-insensitive."""
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[^а-я\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def text_id(text: str) -> str:
    return hashlib.sha256(norm_key(text).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Source: Common Voice
# ---------------------------------------------------------------------------

def fetch_cached(url: str, dest: Path, sha256: str, delay: float = 0) -> bytes:
    if dest.exists():
        data = dest.read_bytes()
    else:
        if delay:
            time.sleep(delay)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    got = sha256_bytes(data)
    if got != sha256:
        raise SystemExit(f"{dest}: sha256 {got} != pinned {sha256}; refusing to build")
    return data


def common_voice_pool() -> list[str]:
    path = TEXT_CACHE / "sentence-collector.txt"
    raw = fetch_cached(CV_URL, path, CV_SHA256)
    pool = []
    seen = set()
    for line in raw.decode("utf-8").splitlines():
        text = clean_text(line)
        if text_problem(text):
            continue
        key = norm_key(text)
        if key in seen:
            continue
        seen.add(key)
        pool.append(text)
    return pool


# ---------------------------------------------------------------------------
# Source: Wikisource (pinned revisions)
# ---------------------------------------------------------------------------

def fetch_wikisource(page: tuple) -> list[str]:
    slug, title, _author, _died, _written, revid, sha = page
    url = "https://ru.wikisource.org/w/api.php?" + urllib.parse.urlencode({
        "action": "parse", "oldid": revid, "prop": "text",
        "format": "json", "formatversion": 2,
    })
    path = WS_CACHE / f"{slug}-{revid}.json"
    data = fetch_cached(url, path, sha, delay=2)
    doc = json.loads(data)
    paras = re.findall(r"<p\b[^>]*>(.*?)</p>", doc["parse"]["text"], re.S)
    out = []
    seen = set()
    for para in paras:
        text = html.unescape(re.sub(r"<[^>]+>", "", para))
        text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁ«»\"—])", text):
            sentence = clean_text(sentence)
            if text_problem(sentence):
                continue
            # dialogue tails such as "— сказал он." are not sentences
            if len(sentence.split()) < 5:
                continue
            # wiki licence/footer boilerplate rendered inside the article
            if sentence.startswith(("Если произведение является", "Текст доступен",
                                    "Последнее изменение", "Материал из")):
                continue
            key = norm_key(sentence)
            if key in seen:
                continue
            seen.add(key)
            out.append(sentence)
    return out


# ---------------------------------------------------------------------------
# Russian numerals
# ---------------------------------------------------------------------------

UNITS_M = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
UNITS_F = ["ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
UNITS_N = ["ноль", "одно", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
         "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят",
        "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
            "шестьсот", "семьсот", "восемьсот", "девятьсот"]


def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n) % 100
    if 11 <= n <= 14:
        return many
    n %= 10
    if n == 1:
        return one
    if 2 <= n <= 4:
        return few
    return many


def cardinal(n: int, gender: str = "m") -> str:
    if n == 0:
        return "ноль"
    if n < 0 or n >= 1_000_000:
        raise ValueError(n)
    parts = []
    if n >= 1000:
        thousands, n = divmod(n, 1000)
        if thousands == 1:
            parts.append("тысяча")
        else:
            parts.append(cardinal(thousands, "f"))
            parts.append(plural(thousands, "тысяча", "тысячи", "тысяч"))
    if n >= 100:
        h, n = divmod(n, 100)
        parts.append(HUNDREDS[h])
    if n >= 20:
        t, n = divmod(n, 10)
        parts.append(TENS[t])
    elif n >= 10:
        parts.append(TEENS[n - 10])
        n = 0
    if n:
        units = {"m": UNITS_M, "f": UNITS_F, "n": UNITS_N}[gender]
        parts.append(units[n])
    return " ".join(parts)


def counted(n: int, one: str, few: str, many: str, gender: str = "m") -> str:
    return f"{cardinal(n, gender)} {plural(n, one, few, many)}"


ORD_NEUTER = [
    "", "первое", "второе", "третье", "четвёртое", "пятое", "шестое", "седьмое",
    "восьмое", "девятое", "десятое", "одиннадцатое", "двенадцатое", "тринадцатое",
    "четырнадцатое", "пятнадцатое", "шестнадцатое", "семнадцатое", "восемнадцатое",
    "девятнадцатое", "двадцатое", "двадцать первое", "двадцать второе",
    "двадцать третье", "двадцать четвёртое", "двадцать пятое", "двадцать шестое",
    "двадцать седьмое", "двадцать восьмое", "двадцать девятое", "тридцатое",
    "тридцать первое",
]


def ordinal_genitive(day: int) -> str:
    word = ORD_NEUTER[day]
    if word.endswith("ое"):
        return word[:-2] + "ого"
    return word[:-2] + "ьего"  # третье -> третьего


MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря"]

# (nominative, accusative, genitive, prepositional)
CITIES = [
    ("Москва", "Москву", "Москвы", "Москве"),
    ("Казань", "Казань", "Казани", "Казани"),
    ("Сочи", "Сочи", "Сочи", "Сочи"),
    ("Санкт-Петербург", "Санкт-Петербург", "Санкт-Петербурга", "Санкт-Петербурге"),
    ("Новосибирск", "Новосибирск", "Новосибирска", "Новосибирске"),
    ("Екатеринбург", "Екатеринбург", "Екатеринбурга", "Екатеринбурге"),
    ("Самара", "Самару", "Самары", "Самаре"),
    ("Омск", "Омск", "Омска", "Омске"),
    ("Тверь", "Тверь", "Твери", "Твери"),
    ("Псков", "Псков", "Пскова", "Пскове"),
    ("Рязань", "Рязань", "Рязани", "Рязани"),
    ("Тула", "Тулу", "Тулы", "Туле"),
    ("Иркутск", "Иркутск", "Иркутска", "Иркутске"),
    ("Воронеж", "Воронеж", "Воронежа", "Воронеже"),
    ("Владимир", "Владимир", "Владимира", "Владимире"),
]

# nominative, genitive, dative, instrumental
NAMES = [
    ("Иван", "Ивана", "Ивану", "Иваном", "m"),
    ("Мария", "Марии", "Марии", "Марией", "f"),
    ("Олег", "Олега", "Олегу", "Олегом", "m"),
    ("Анна", "Анны", "Анне", "Анной", "f"),
    ("Пётр", "Петра", "Петру", "Петром", "m"),
    ("Софья", "Софьи", "Софье", "Софьей", "f"),
    ("Дмитрий", "Дмитрия", "Дмитрию", "Дмитрием", "m"),
    ("Елена", "Елены", "Елене", "Еленой", "f"),
    ("Сергей", "Сергея", "Сергею", "Сергеем", "m"),
    ("Ольга", "Ольги", "Ольге", "Ольгой", "f"),
    ("Алексей", "Алексея", "Алексею", "Алексеем", "m"),
    ("Татьяна", "Татьяны", "Татьяне", "Татьяной", "f"),
    ("Михаил", "Михаила", "Михаилу", "Михаилом", "m"),
    ("Ирина", "Ирины", "Ирине", "Ириной", "f"),
    ("Николай", "Николая", "Николаю", "Николаем", "m"),
    ("Наталья", "Натальи", "Наталье", "Натальей", "f"),
    ("Павел", "Павла", "Павлу", "Павлом", "m"),
    ("Ксения", "Ксении", "Ксении", "Ксенией", "f"),
    ("Андрей", "Андрея", "Андрею", "Андреем", "m"),
    ("Вера", "Веры", "Вере", "Верой", "f"),
    ("Максим", "Максима", "Максиму", "Максимом", "m"),
    ("Юлия", "Юлии", "Юлии", "Юлией", "f"),
    ("Борис", "Бориса", "Борису", "Борисом", "m"),
    ("Нина", "Нины", "Нине", "Ниной", "f"),
    ("Артём", "Артёма", "Артёму", "Артёмом", "m"),
    ("Дарья", "Дарьи", "Дарье", "Дарьей", "f"),
]
SURNAMES = [("Петров", "Петрова"), ("Смирнов", "Смирнова"), ("Кузнецов", "Кузнецова"),
            ("Соколов", "Соколова"), ("Попов", "Попова"), ("Лебедев", "Лебедева"),
            ("Новиков", "Новикова"), ("Морозов", "Морозова"), ("Волков", "Волкова"),
            ("Соловьёв", "Соловьёва")]


# ---------------------------------------------------------------------------
# Template generators
# ---------------------------------------------------------------------------

def gen_numbers() -> list[str]:
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 23, 25,
              27, 28, 30, 32, 35, 37, 40, 42, 45, 48, 50, 52, 55, 58, 60, 63, 65, 67,
              70, 72, 75, 78, 80, 83, 85, 88, 90, 92, 95, 97, 100, 102, 105, 108, 110,
              112, 115, 120, 123, 125, 130, 135, 140, 145, 150, 160, 175, 180, 200, 205,
              210, 220, 230, 240, 250, 260, 275, 300, 315, 320, 350, 400, 420, 450, 500,
              550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1200, 1500, 1800,
              2000, 2500, 3000, 4500, 5000, 7000]
    out = []
    for n in values:
        out += [
            f"На столе лежало {counted(n, 'яблоко', 'яблока', 'яблок', 'n')}.",
            f"В зале собралось {counted(n, 'человек', 'человека', 'человек')}.",
            f"Мы прошли {counted(n, 'километр', 'километра', 'километров')} по лесу.",
            f"Он прочитал {counted(n, 'страница', 'страницы', 'страниц', 'f')} за вечер.",
            f"В коробке осталось {counted(n, 'карандаш', 'карандаша', 'карандашей')}.",
            f"До города оставалось {counted(n, 'километр', 'километра', 'километров')}.",
            f"Она купила {counted(n, 'тетрадь', 'тетради', 'тетрадей', 'f')} в магазине.",
            f"{counted(n, 'ученик', 'ученика', 'учеников')} сдали экзамен с первого раза.",
            f"На складе хранится {counted(n, 'коробка', 'коробки', 'коробок', 'f')}.",
            f"Поезд опоздал на {counted(n, 'минута', 'минуты', 'минут', 'f')}.",
            f"В этом доме {counted(n, 'квартира', 'квартиры', 'квартир', 'f')}.",
            f"У нас в команде {counted(n, 'человек', 'человека', 'человек')}.",
            f"Я ждал {counted(n, 'минута', 'минуты', 'минут', 'f')} под дождём.",
            f"Он сделал {counted(n, 'шаг', 'шага', 'шагов')} и остановился.",
            f"В книге {counted(n, 'страница', 'страницы', 'страниц', 'f')}.",
            f"Через {counted(n, 'день', 'дня', 'дней')} начнётся отпуск.",
            f"Она повторила это {counted(n, 'раз', 'раза', 'раз')}.",
            f"В зале стояло {counted(n, 'стол', 'стола', 'столов')}.",
        ]
    return out


def gen_dates() -> list[str]:
    out = []
    for month in range(1, 13):
        for day in range(1, 29):
            gen = f"{ordinal_genitive(day)} {MONTHS_GEN[month - 1]}"
            acc = f"{ORD_NEUTER[day]} {MONTHS_GEN[month - 1]}"
            name = NAMES[(day + month) % len(NAMES)][0]
            city = CITIES[(day * 3 + month) % len(CITIES)][1]
            out += [
                f"{gen} {name} поедет в {city}.",
                f"Встреча назначена на {acc}.",
                f"Мы вернёмся {gen}.",
                f"До {gen} нужно сдать документы.",
                f"С {gen} магазин работает по новому графику.",
                f"Экзамен перенесли на {acc}.",
                f"{gen} начинается новый учебный год.",
            ]
    return out


def gen_times() -> list[str]:
    out = []
    hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    minutes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    for h in hours:
        for m in minutes:
            hw = "час" if h == 1 else cardinal(h)
            mw = cardinal(m, "f")
            out += [
                f"Автобус отправляется в {hw} {mw}.",
                f"Электричка уходит в {hw} {mw} с третьей платформы.",
                f"Магазин закрывается в {hw} вечера.",
                f"Созвон назначен на {hw} часов утра.",
                f"Я освобожусь после {hw} часов.",
                f"Поезд прибывает в {hw} часов {counted(m, 'минута', 'минуты', 'минут', 'f')}."
                if m else f"Поезд прибывает ровно в {hw} часов.",
            ]
    return out


def gen_money() -> list[str]:
    values = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 90, 100, 110, 120,
              125, 130, 140, 150, 160, 175, 180, 190, 200, 210, 220, 230, 240, 250, 260,
              275, 280, 300, 320, 350, 380, 400, 420, 450, 480, 500, 550, 600, 650, 700,
              750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1800]
    out = []
    for i, n in enumerate(values):
        k = [0, 5, 10, 15, 20, 25, 30, 40, 50, 55, 60, 75, 80, 90, 99][i % 15]
        rub = counted(n, "рубль", "рубля", "рублей")
        kop = counted(k, "копейка", "копейки", "копеек")
        out += [
            f"Это стоит {rub} {kop}.",
            f"Билет стоит {rub}.",
            f"Я заплатил {rub} за обед.",
            f"На счету осталось {rub} {kop}.",
            f"Он взял с собой {rub} мелочью.",
            f"Мы отложили на отпуск {rub}.",
            f"Мы собрали {rub} на подарок.",
            f"Кофе стоит {rub}, а чай — {values[(i + 9) % len(values)]} рублей.",
        ]
    return out


def gen_names() -> list[str]:
    out = []
    for i, (nom, gen, dat, ins, gender) in enumerate(NAMES):
        nom2, _gen2, dat2, _ins2, _g2 = NAMES[(i + 7) % len(NAMES)]
        sur_m, sur_f = SURNAMES[i % len(SURNAMES)]
        surname = sur_f if gender == "f" else sur_m
        verb = "набрала" if gender == "f" else "набрал"
        city = CITIES[i % len(CITIES)]
        out += [
            f"Привет, {nom}! Как прошёл твой день?",
            f"{nom}, позвони {dat2} до семи часов.",
            f"Встреча с {ins} перенесена на среду.",
            f"Меня зовут {nom} {surname}, я живу в {city[3]}.",
            f"{nom} и {nom2} поедут в {city[1]}.",
            f"Кто сказал {dat}, что магазин закрывается в девять?",
            f"Подожди, ты хочешь сказать, что {nom} ушёл в семь и до сих пор не вернулся?",
            f"{nom} {surname} {verb} восемьдесят процентов на экзамене.",
            f"Я давно не видел {gen}.",
            f"Передай, пожалуйста, {dat} эту книгу.",
            f"{nom} живёт на пятом этаже, а {nom2} — на втором.",
            f"Спроси у {gen}, когда начинается собрание.",
            f"{nom} и {nom2} живут в одном доме.",
            f"Мы часто видимся с {ins}.",
            f"У {gen} есть старшая сестра.",
            f"Работа {gen} начинается в девять утра.",
            f"Я передал документы {dat} вчера.",
            f"Телефон {gen} не отвечает уже час.",
        ]
    return out


def gen_questions() -> list[str]:
    out = []
    for nom, _gen, dat, ins, _g in NAMES:
        out += [
            f"Ты уже позвонил {dat}?",
            f"Почему {nom} не отвечает на звонки?",
            f"Кто звонил {dat} сегодня утром?",
            f"Ты знаешь, где сейчас {nom}?",
            f"Ты не знаешь, где живёт {nom}?",
            f"Сколько лет {dat} исполнилось?",
            f"Ты давно знаком с {ins}?",
            f"Что {nom} думает о нашем плане?",
            f"Куда {nom} собирается в отпуск?",
            f"Когда {nom} вернётся из поездки?",
            f"Ты передал {dat} мою просьбу?",
            f"Почему {nom} так редко звонит?",
        ]
    for city in CITIES:
        out += [
            f"Ты уже купил билет в {city[1]}?",
            f"Сколько стоит билет на поезд до {city[2]}?",
            f"Когда ты вернёшься из {city[2]}?",
            f"Сколько времени займёт дорога до {city[2]}?",
            f"Ты был в {city[3]} раньше?",
            f"Почему ты не хочешь поехать в {city[1]}?",
            f"Кто поедет с нами в {city[1]}?",
            f"Что посмотреть в {city[3]} за один день?",
        ]
    out += [
        "Ты не подскажешь, который сейчас час?",
        "Мы успеем на электричку в девять сорок?",
        "Ты правда прочитал все двадцать три главы?",
        "Ты не знаешь, где находится библиотека?",
        "Что ты будешь делать в субботу вечером?",
        "Можно я возьму твою книгу на неделю?",
        "Нам нужно купить продукты до закрытия магазина?",
        "Ты помнишь, во сколько начинается фильм?",
        "Ты уже решил, что возьмёшь с собой?",
        "Сколько времени ты добираешься до работы?",
        "Куда пропали все чистые полотенца?",
        "Почему на кухне опять горит свет?",
        "Ты закрыл дверь на нижний замок?",
        "Кто оставил посуду в раковине?",
        "Сколько сахара ты кладёшь в чай?",
        "Ты уже выключил компьютер?",
        "Почему ты не взял с собой зонт?",
        "Куда ты поставил коробку с книгами?",
        "Ты сможешь зайти завтра после обеда?",
        "Сколько дней осталось до отпуска?",
    ]
    return out


def gen_lists() -> list[str]:
    items = ["молоко", "хлеб", "сыр", "масло", "яйца", "картошку", "морковь", "лук",
             "сахар", "соль", "чай", "кофе", "печенье", "яблоки", "бананы", "сметану",
             "творог", "рис", "макароны", "сок", "рыбу", "курицу", "овощи", "фрукты"]
    items_nom = ["молоко", "хлеб", "сыр", "масло", "яйца", "картошка", "морковь", "лук",
                 "сахар", "соль", "чай", "кофе", "печенье", "яблоки", "бананы", "сметана",
                 "творог", "рис", "макароны", "сок", "рыба", "курица", "овощи", "фрукты"]
    places = [("в", "аптеку"), ("в", "банк"), ("в", "библиотеку"), ("в", "магазин"),
              ("на", "почту"), ("в", "поликлинику"), ("на", "вокзал"), ("на", "рынок"),
              ("в", "школу"), ("в", "офис"), ("в", "кафе"), ("в", "музей"),
              ("в", "театр"), ("в", "парк"), ("на", "стадион")]
    rng = random.Random(f"{SEED}:lists")
    out = []
    for i in range(240):
        ia, ib, ic = rng.sample(range(len(items)), 3)
        p1, p2, p3 = rng.sample(places, 3)
        out.append([
            f"Купи, пожалуйста, {items[ia]}, {items[ib]} и {items[ic]}.",
            f"Сначала мы заедем {p1[0]} {p1[1]}, потом {p2[0]} {p2[1]}, а в конце — {p3[0]} {p3[1]}.",
            f"На столе были {items_nom[ia]}, {items_nom[ib]} и {items_nom[ic]}.",
            f"Возьми с собой {items[ia]}, {items[ib]} и {items[ic]}.",
            f"В сумке лежали {items_nom[ia]}, {items_nom[ib]} и {items_nom[ic]}.",
        ][i % 5])
    return out


def gen_abbreviations() -> list[str]:
    """Stratified abbreviations: 15 rows per pattern so every form is exercised."""
    libs = ["Пушкина", "Гоголя", "Чехова", "Лермонтова", "Тургенева"]
    closing = ["шести", "семи", "восьми", "девяти", "десяти"]
    ordinals = ["второй", "третий", "четвёртый", "пятый", "шестой",
                "седьмой", "восьмой", "девятый", "десятый", "одиннадцатый",
                "двенадцатый", "тринадцатый", "четырнадцатый", "пятнадцатый",
                "шестнадцатый"]
    streets = ["Пушкина", "Гоголя", "Чехова", "Садовую", "Лесную",
               "Школьную", "Центральную", "Речную", "Полевую", "Зелёную",
               "Солнечную", "Мира", "Лермонтова", "Тургенева", "Гоголевскую"]
    possible = ["вполне допустимо", "совсем не сложно", "в общем-то реально",
                "легко проверить", "давно известно", "вполне вероятно",
                "не так уж важно", "совершенно безопасно", "вполне объяснимо",
                "почти всегда так", "вовсе не обязательно", "очень может быть",
                "в целом нормально", "абсолютно верно", "достаточно просто"]
    pools: dict[str, list[str]] = {k: [] for k in (
        "page", "address", "price", "print", "library", "rule",
        "possible", "send", "see", "metro", "rub", "kv")}
    for i in range(1, 61):
        n = i * 3
        pools["page"].append(f"См. примечание на странице {cardinal(n)}.")
        pools["address"].append(f"Он живёт на ул. Пушкина, дом {cardinal(n)}, квартира пять.")
        pools["price"].append(
            f"Это стоит {counted(n, 'рубль', 'рубля', 'рублей')} "
            f"и {counted(i, 'копейка', 'копейки', 'копеек', 'f')}.")
        pools["print"].append(
            f"Книга издана в г. Москве, тираж "
            f"{counted(n * 100, 'экземпляр', 'экземпляра', 'экземпляров')}.")
        pools["library"].append(f"Библиотека им. {libs[i % 5]} работает до {closing[(i // 5) % 5]} вечера.")
        pools["rule"].append(f"Сначала прочитайте правило, т. е. {ordinals[i % len(ordinals)]} параграф.")
        pools["possible"].append(f"Это возможно, т. е. {possible[i % 15]}.")
        pools["send"].append(f"Пришли мне адрес, т. е. улицу {streets[i % 15]} и номер дома.")
        pools["see"].append(f"Смотри с. {cardinal(n)}, там есть таблица.")
        pools["metro"].append(f"Мы встретимся у метро, т. е. у выхода номер {cardinal(i)}.")
        pools["rub"].append(
            f"Билет стоит {cardinal(n * 10)} руб., а детский — {cardinal(n * 5)} руб.")
        pools["kv"].append(f"Он живёт на ул. Пушкина, д. {cardinal(n)}, кв. пять.")
    out = []
    base, extra = divmod(150, len(pools))
    for idx, (kind, pool) in enumerate(pools.items()):
        rng = random.Random(f"{SEED}:abbr:{kind}")
        rng.shuffle(pool)
        out += list(dict.fromkeys(pool))[:base + (1 if idx < extra else 0)]
    return out


TEMPLATE_GROUPS = [
    ("number", gen_numbers, 800),
    ("date", gen_dates, 250),
    ("date", gen_times, 200),
    ("money", gen_money, 300),
    ("name", gen_names, 400),
    ("question", gen_questions, 350),
    ("template", gen_lists, 150),
    ("template", gen_abbreviations, 150),
]


def pick(candidates: list[str], count: int, salt: str) -> list[str]:
    seen = set()
    uniq = []
    for text in candidates:
        key = norm_key(text)
        if key in seen or text_problem(text):
            continue
        seen.add(key)
        uniq.append(text)
    uniq.sort()
    rng = random.Random(f"{SEED}:{salt}")
    rng.shuffle(uniq)
    if len(uniq) < count:
        raise SystemExit(f"{salt}: only {len(uniq)} usable candidates, need {count}")
    return uniq[:count]


# ---------------------------------------------------------------------------
# Own sentences
# ---------------------------------------------------------------------------

def read_own(batch: int) -> list[str]:
    path = OWN_DIR / f"batch{batch}.txt"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = clean_text(line)
        if not text or text.startswith("#"):
            continue
        problem = text_problem(text)
        if problem:
            raise SystemExit(f"{path}: {problem}: {text!r}")
        out.append(text)
    return out


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

def make_row(text: str, split: str, category: str, origin: str, source_id: str,
             license_: str, source_url: str, batch: int) -> dict:
    return {
        "id": text_id(text),
        "text": text,
        "split": split,
        "category": category,
        "origin": origin,
        "source_id": source_id,
        "source_license": license_,
        "source_url": source_url,
        "word_count": len(text.split()),
        "char_count": len(text),
        "batch": batch,
    }


def assign_splits(rows: list[dict]) -> None:
    ids = sorted(row["id"] for row in rows)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(ids)
    n_val = round(len(ids) * VAL_PER_1000 / 1000)
    val = set(ids[:n_val])
    test = set(ids[n_val:2 * n_val])
    for row in rows:
        row["split"] = "val" if row["id"] in val else "test" if row["id"] in test else "train"


def dedupe(rows: list[dict]) -> list[dict]:
    """Keep the first occurrence of each normalised text; ids must be unique."""
    seen = set()
    out = []
    for row in rows:
        key = norm_key(row["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build() -> None:
    rows: list[dict] = []

    cv = common_voice_pool()
    rng = random.Random(f"{SEED}:cv")
    for text in rng.sample(sorted(cv), 3000):
        rows.append(make_row(text, "", "conversation", "common_voice_sentence_collector",
                             "cv-ru-sentence-collector", CV_LICENSE, CV_URL, 1))

    ws_candidates = []
    ws_seen = set()
    for page in WIKISOURCE_PAGES:
        slug, title, author, _died, _written, revid, _sha = page
        url = f"https://ru.wikisource.org/w/index.php?oldid={revid}"
        for text in fetch_wikisource(page):
            key = norm_key(text)
            if key in ws_seen:
                continue
            ws_seen.add(key)
            ws_candidates.append((text, title, author, url))
    rng = random.Random(f"{SEED}:wikisource")
    rng.shuffle(ws_candidates)
    for text, title, author, url in ws_candidates[:900]:
        rows.append(make_row(text, "", "reading", "wikisource_pd",
                             f"wikisource:{title}@{url.rsplit('=', 1)[1]}",
                             "public-domain", url, 1))

    for text in read_own(1):
        category = "question" if text.endswith("?") else "conversation"
        rows.append(make_row(text, "", category, "project_llm", "own-batch1",
                             "cc0-1.0", "project-own", 1))

    rows = dedupe(rows)
    if len(rows) != 4500:
        raise SystemExit(f"batch 1 has {len(rows)} unique sentences, expected 4500")
    assign_splits(rows)
    rows.sort(key=lambda r: r["id"])

    batch2: list[dict] = []
    for category, generator, count in TEMPLATE_GROUPS:
        for text in pick(generator(), count, category + generator.__name__):
            batch2.append(make_row(text, "train", category, "project_templates",
                                   f"template-{category}", "cc0-1.0", "project-own", 2))
    for text in read_own(2):
        batch2.append(make_row(text, "train", own_category(text), "project_llm",
                               "own-batch2", "cc0-1.0", "project-own", 2))
    batch2 = dedupe(batch2)
    if len(batch2) != 3500:
        raise SystemExit(f"batch 2 has {len(batch2)} unique sentences, expected 3500")
    batch2.sort(key=lambda r: r["id"])

    ids = {r["id"] for r in rows} | {r["id"] for r in batch2}
    if len(ids) != 8000:
        raise SystemExit("duplicate ids across batches")

    write_jsonl(OUT_DIR / "batch1.jsonl", rows)
    write_jsonl(OUT_DIR / "batch2.jsonl", batch2)
    write_splits(rows + batch2)
    report(rows + batch2)


def own_category(text: str) -> str:
    """Assign a category to a hand-written sentence from its content."""
    low = text.lower()
    if re.search(r"рубл|копейк|доллар|евро|скидк|стоит|дороже|дешевле|заплат", low):
        return "money"
    if re.search(r"январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр|"
                 r"понедельник|вторник|сред|четверг|пятниц|суббот|воскресень", low):
        return "date"
    if re.search(r"\bчас|минут", low):
        return "date"
    if re.search(r"\b(один|одна|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|"
                 r"одиннадцать|двенадцать|тринадцать|четырнадцать|пятнадцать|шестнадцать|"
                 r"семнадцать|восемнадцать|девятнадцать|двадцать|тридцать|сорок|пятьдесят|"
                 r"шестьдесят|семьдесят|восемьдесят|девяносто|сто|двести|триста|четыреста|"
                 r"пятьсот|тысяч)", low):
        return "number"
    if text.endswith("?"):
        return "question"
    if re.search(r"\b(Иван|Мария|Олег|Анна|Пётр|Софья|Дмитрий|Елена|Сергей|Ольга|Алексей|"
                 r"Татьяна|Михаил|Ирина|Николай|Наталья|Павел|Ксения|Андрей|Вера|Максим|"
                 r"Юлия|Борис|Нина|Артём|Дарья)", text):
        return "name"
    return "conversation"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_splits(rows: list[dict]) -> None:
    batch1 = (OUT_DIR / "batch1.jsonl").read_bytes()
    batch2 = (OUT_DIR / "batch2.jsonl").read_bytes()
    manifest = "\n".join(f"{row['id']} {row['split']}" for row in sorted(rows, key=lambda r: r["id"]))
    counts = {}
    for row in rows:
        counts[(row["batch"], row["split"])] = counts.get((row["batch"], row["split"]), 0) + 1
    lines = [
        "# Frozen split manifest for the F2 Russian script corpus.",
        "# Regenerate with: python3 scripts/text/build_ru_scripts.py",
        f"batch1.jsonl sha256:{sha256_bytes(batch1)}",
        f"batch2.jsonl sha256:{sha256_bytes(batch2)}",
        f"split-manifest sha256:{sha256_bytes(manifest.encode())}",
        " ".join(f"batch{b}-{s}={counts.get((b, s), 0)}"
                 for b in (1, 2) for s in ("train", "val", "test")),
    ]
    (OUT_DIR / "SPLITS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(rows: list[dict]) -> None:
    print(f"sentences: {len(rows)}")
    for batch in (1, 2):
        chunk = [r for r in rows if r["batch"] == batch]
        print(f"batch{batch}: {len(chunk)}")
        for split in ("train", "val", "test"):
            print(f"  {split}: {sum(1 for r in chunk if r['split'] == split)}")
    print("by category:")
    cats = {}
    for row in rows:
        cats[row["category"]] = cats.get(row["category"], 0) + 1
    for cat, n in sorted(cats.items()):
        print(f"  {cat}: {n}")
    print("by origin:")
    origins = {}
    for row in rows:
        origins[row["origin"]] = origins.get(row["origin"], 0) + 1
    for origin, n in sorted(origins.items()):
        print(f"  {origin}: {n}")


if __name__ == "__main__":
    build()

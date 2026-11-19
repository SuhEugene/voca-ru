//! Deterministic Russian text normalisation for the phoneme path.
//!
//! Direct port of `training/voca_ru/text/normalize.py`. Same rule order, same
//! outputs; the parity harness compares this against the Python module.

use std::sync::OnceLock;

use fancy_regex::{Captures, Regex};
use unicode_normalization::UnicodeNormalization;

use crate::numbers;

pub const MONTHS_GEN: [&str; 12] = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
];

const LATIN_NAMES: [(char, &str); 26] = [
    ('a', "эй"),
    ('b', "би"),
    ('c', "си"),
    ('d', "ди"),
    ('e', "и"),
    ('f', "эф"),
    ('g', "джи"),
    ('h', "эйч"),
    ('i', "ай"),
    ('j', "джей"),
    ('k', "кей"),
    ('l', "эл"),
    ('m', "эм"),
    ('n', "эн"),
    ('o', "о"),
    ('p', "пи"),
    ('q', "кью"),
    ('r', "ар"),
    ('s', "эс"),
    ('t', "ти"),
    ('u', "ю"),
    ('v', "ви"),
    ('w', "дабл ю"),
    ('x', "экс"),
    ('y', "уай"),
    ('z', "зет"),
];

const UNITS: [(&str, (&str, &str, &str)); 12] = [
    ("км", ("километр", "километра", "километров")),
    ("м", ("метр", "метра", "метров")),
    ("см", ("сантиметр", "сантиметра", "сантиметров")),
    ("мм", ("миллиметр", "миллиметра", "миллиметров")),
    ("кг", ("килограмм", "килограмма", "килограммов")),
    ("г", ("грамм", "грамма", "граммов")),
    ("мг", ("миллиграмм", "миллиграмма", "миллиграммов")),
    ("л", ("литр", "литра", "литров")),
    ("мл", ("миллилитр", "миллилитра", "миллилитров")),
    ("га", ("гектар", "гектара", "гектаров")),
    ("руб", ("рубль", "рубля", "рублей")),
    ("коп", ("копейка", "копейки", "копеек")),
];

const DIGIT_WORDS: [&str; 10] = [
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
];

const GEN_PREPS: [&str; 15] = [
    "до",
    "от",
    "из",
    "без",
    "около",
    "для",
    "у",
    "после",
    "кроме",
    "ради",
    "больше",
    "меньше",
    "свыше",
    "порядка",
    "числе",
];
const DAT_PREPS: [&str; 2] = ["к", "по"];
const PREP_PREPS: [&str; 5] = ["в", "на", "о", "об", "при"];
const INS_PREPS: [&str; 4] = ["с", "между", "над", "под"];

fn rx(pattern: &str) -> Regex {
    Regex::new(pattern).expect("static regex")
}

fn sub(text: &str, pattern: &str, repl: impl FnMut(&Captures<'_>) -> String) -> String {
    let re = rx(pattern);
    re.replace_all(text, repl).into_owned()
}

fn clean(text: &str) -> String {
    let mut out: String = text.nfc().collect();
    for ch in ['\u{00a0}', '\u{202f}', '\u{2009}'] {
        out = out.replace(ch, " ");
    }
    out = out.replace('\u{0301}', "");
    out = out.to_lowercase();
    for bad in [
        '\u{200b}', '\u{200c}', '\u{200d}', '\u{feff}', '\t', '\r', '\n',
    ] {
        out = out.replace(bad, " ");
    }
    out
}

fn merge_thousands(text: &str) -> String {
    sub(text, r"(?<!\d)(\d{1,3})(?:[ ]\d{3})+(?!\d)", |caps| {
        caps.get(0).unwrap().as_str().replace(' ', "")
    })
}

struct ScaleUnit {
    key: &'static str,
    one: &'static str,
    few: &'static str,
    many: &'static str,
    gender: numbers::Gender,
}

const SCALE_UNITS: [ScaleUnit; 4] = [
    ScaleUnit {
        key: "тыс.",
        one: "тысяча",
        few: "тысячи",
        many: "тысяч",
        gender: numbers::F,
    },
    ScaleUnit {
        key: "тыс",
        one: "тысяча",
        few: "тысячи",
        many: "тысяч",
        gender: numbers::F,
    },
    ScaleUnit {
        key: "млн",
        one: "миллион",
        few: "миллиона",
        many: "миллионов",
        gender: numbers::M,
    },
    ScaleUnit {
        key: "млрд",
        one: "миллиард",
        few: "миллиарда",
        many: "миллиардов",
        gender: numbers::M,
    },
];

fn expand_scales(text: &str) -> String {
    let re = rx(r"(?<!\d)(\d{1,9})\s*(тыс\.|тыс\b|млн\b|млрд\b)");
    re.replace_all(text, |caps: &Captures<'_>| {
        let caps = caps;
        let value: i64 = caps.get(1).unwrap().as_str().parse().unwrap();
        let unit = caps.get(2).unwrap().as_str();
        let scale = SCALE_UNITS.iter().find(|s| s.key == unit).unwrap();
        let mut words = numbers::cardinal(value, numbers::NOM, scale.gender);
        words.push(numbers::agree(value, scale.one, scale.few, scale.many));
        words.join(" ")
    })
    .into_owned()
}

fn expand_abbreviations(text: &str) -> String {
    let rules: [(&str, &str); 24] = [
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
    ];
    let mut out = text.to_string();
    for (pattern, repl) in rules {
        out = rx(pattern).replace_all(&out, repl).into_owned();
    }
    out = sub(&out, r"(?<=\d)\s+г\.", |_| " года".into());
    out = sub(&out, r"(?<=\d)\s+с\.", |_| " страница".into());
    out = sub(&out, r"(?<=\d)\s+д\.", |_| " дом".into());
    out = sub(&out, r"\bг\.", |_| "город".into());
    out = sub(&out, r"\bс\.", |_| "страница".into());
    out = sub(&out, r"\bд\.", |_| "дом".into());
    out
}

const MASCULINE_GENITIVES: [&str; 13] = [
    "часа",
    "дня",
    "года",
    "месяца",
    "рубля",
    "метра",
    "километра",
    "литра",
    "процента",
    "килограмма",
    "грамма",
    "доллара",
    "миллиона",
];

const TRIM: [char; 19] = [
    '.', ',', '!', '?', ';', ':', '«', '»', '"', '\'', '(', ')', '[', ']', '{', '}', '—', '–', '…',
];

fn trim_word(word: &str) -> String {
    word.trim_matches(|c: char| TRIM.contains(&c)).to_string()
}

fn gender_of(word: &str) -> numbers::Gender {
    let word = trim_word(word);
    if word.is_empty() {
        return numbers::M;
    }
    if MASCULINE_GENITIVES.contains(&word.as_str()) {
        return numbers::M;
    }
    let ends = |suffixes: &[&str]| suffixes.iter().any(|s| word.ends_with(s));
    if ends(&["ая", "яя"]) {
        return numbers::F;
    }
    if ends(&["ое", "ее", "мя"]) {
        return numbers::N;
    }
    if ends(&["а", "я", "ы", "и"]) {
        return numbers::F;
    }
    if ends(&["о", "е"]) {
        return numbers::N;
    }
    numbers::M
}

fn nearest_word_before(text: &str, pos: usize) -> String {
    let head = text[..pos].trim_end();
    if head.is_empty() {
        return String::new();
    }
    head.split_whitespace()
        .last()
        .map(|w| trim_word(w).to_lowercase())
        .unwrap_or_default()
}

fn word_after(text: &str, pos: usize) -> String {
    let rest = text[pos..].trim_start();
    static WORD: OnceLock<Regex> = OnceLock::new();
    let re = WORD.get_or_init(|| rx(r"^[а-яё-]+"));
    re.find(rest)
        .ok()
        .flatten()
        .map(|m| m.as_str().to_string())
        .unwrap_or_default()
}

fn case_for(text: &str, pos: usize, lookahead: bool, after: Option<usize>) -> numbers::Case {
    let prep = nearest_word_before(text, pos);
    if GEN_PREPS.contains(&prep.as_str()) {
        return numbers::GEN;
    }
    if DAT_PREPS.contains(&prep.as_str()) {
        return numbers::DAT;
    }
    if INS_PREPS.contains(&prep.as_str()) {
        return numbers::INS;
    }
    if PREP_PREPS.contains(&prep.as_str()) {
        let nxt = if lookahead {
            word_after(text, after.unwrap_or(pos))
        } else {
            String::new()
        };
        if (prep == "в" || prep == "на")
            && !(nxt.ends_with("ах")
                || nxt.ends_with("ях")
                || nxt.ends_with('е')
                || nxt.ends_with('и'))
        {
            return numbers::ACC;
        }
        return numbers::PREP;
    }
    numbers::NOM
}

fn day_ordinal(day: &str) -> String {
    numbers::ordinal(day.parse().unwrap_or(0), numbers::GEN, numbers::M)
}

fn year_ordinal(year: &str, case: numbers::Case) -> String {
    numbers::ordinal(year.parse().unwrap_or(0), case, numbers::M)
}

fn expand_dates(text: &str) -> String {
    let mut out = sub(text, r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", |caps| {
        let caps = caps;
        let day = caps.get(1).unwrap().as_str();
        let month: i64 = caps.get(2).unwrap().as_str().parse().unwrap();
        let year = caps.get(3).unwrap().as_str();
        if !(1..=12).contains(&month) {
            return caps.get(0).unwrap().as_str().to_string();
        }
        format!(
            "{} {} {} года",
            day_ordinal(day),
            MONTHS_GEN[(month - 1) as usize],
            year_ordinal(year, numbers::GEN)
        )
    });
    let months = MONTHS_GEN.join("|");
    let pattern =
        format!(r"\b(\d{{1,2}})\s+({months})(?:\s+(\d{{4}})(?:\s+(?:года|году|год))?)?\b");
    out = sub(&out, &pattern, |caps| {
        let caps = caps;
        let day = caps.get(1).unwrap().as_str();
        let month = caps.get(2).unwrap().as_str();
        let year = caps.get(3).map(|m| m.as_str());
        let mut out = format!("{} {}", day_ordinal(day), month);
        if let Some(year) = year {
            out.push_str(&format!(" {} года", year_ordinal(year, numbers::GEN)));
        }
        out
    });
    out
}

fn expand_time(text: &str) -> String {
    sub(text, r"\b(\d{1,2}):(\d{2})\b", |caps| {
        let caps = caps;
        let hour: i64 = caps.get(1).unwrap().as_str().parse().unwrap();
        let minute = caps.get(2).unwrap().as_str();
        if hour > 23 || minute.parse::<i64>().unwrap_or(99) > 59 {
            return caps.get(0).unwrap().as_str().to_string();
        }
        let mut words = numbers::cardinal(hour, numbers::NOM, numbers::M);
        if minute == "00" {
            words.push("ноль".into());
            words.push("ноль".into());
        } else if minute.starts_with('0') {
            words.push("ноль".into());
            words.extend(numbers::cardinal(
                minute.parse().unwrap(),
                numbers::NOM,
                numbers::M,
            ));
        } else {
            words.extend(numbers::cardinal(
                minute.parse().unwrap(),
                numbers::NOM,
                numbers::M,
            ));
        }
        words.join(" ")
    })
}

fn expand_money(text: &str) -> String {
    let mut out = sub(
        text,
        r"(?<!\d)(\d{1,9})\s*(?:₽|руб(?:\.|ль|ля|лей)?\b)",
        |caps| {
            let caps = caps;
            let value: i64 = caps.get(1).unwrap().as_str().parse().unwrap();
            let case = case_for(text, caps.get(0).unwrap().start(), false, None);
            let mut words = numbers::cardinal(value, case, numbers::M);
            words.push(numbers::agree(value, "рубль", "рубля", "рублей"));
            words.join(" ")
        },
    );
    out = sub(
        &out,
        r"(?<!\d)(\d{1,9})\s*(?:коп\.?|копейка|копейки|копеек)\b",
        |caps| {
            let caps = caps;
            let value: i64 = caps.get(1).unwrap().as_str().parse().unwrap();
            let case = case_for(&out, caps.get(0).unwrap().start(), false, None);
            let mut words = numbers::cardinal(value, case, numbers::F);
            words.push(numbers::agree(value, "копейка", "копейки", "копеек"));
            words.join(" ")
        },
    );
    out = sub(&out, r"(?<!\d)(\d{1,9})\s*€", |caps| {
        let caps = caps;
        let value: i64 = caps.get(1).unwrap().as_str().parse().unwrap();
        let mut words = numbers::cardinal(value, numbers::NOM, numbers::M);
        words.push("евро".into());
        words.join(" ")
    });
    sub(&out, r"(?<!\d)(\d{1,9})\s*\$", |caps| {
        let caps = caps;
        let value: i64 = caps.get(1).unwrap().as_str().parse().unwrap();
        let mut words = numbers::cardinal(value, numbers::NOM, numbers::M);
        words.push("долларов".into());
        words.join(" ")
    })
}

fn expand_percent(text: &str) -> String {
    sub(text, r"(?<!\d)(\d{1,9})\s*%", |caps| {
        let caps = caps;
        let value: i64 = caps.get(1).unwrap().as_str().parse().unwrap();
        let mut words = numbers::cardinal(value, numbers::NOM, numbers::M);
        words.push(numbers::agree(value, "процент", "процента", "процентов"));
        words.join(" ")
    })
}

fn expand_ordinals(text: &str) -> String {
    sub(
        text,
        r"(?<!\d)(\d{1,4})-?(й|ый|ой|го|му|м|х|я|ю|е|ая|ое)\b",
        |caps| {
            let caps = caps;
            let value: i64 = caps.get(1).unwrap().as_str().parse().unwrap();
            let suffix = caps.get(2).unwrap().as_str();
            let (case, gender) = match suffix {
                "й" | "ый" | "ой" => (numbers::NOM, numbers::M),
                "го" => (numbers::GEN, numbers::M),
                "му" => (numbers::DAT, numbers::M),
                "м" => (numbers::INS, numbers::M),
                "х" => (numbers::PREP, numbers::M),
                "я" => (numbers::NOM, numbers::F),
                "ю" => (numbers::ACC, numbers::F),
                "е" => (numbers::NOM, numbers::N),
                "ая" => (numbers::NOM, numbers::F),
                "ое" => (numbers::NOM, numbers::N),
                _ => (numbers::NOM, numbers::M),
            };
            numbers::ordinal(value, case, gender)
        },
    )
}

fn expand_years(text: &str) -> String {
    sub(text, r"\b(\d{4})\s+(году|года|год)\b", |caps| {
        let caps = caps;
        let year = caps.get(1).unwrap().as_str();
        let suffix = caps.get(2).unwrap().as_str();
        let case = if suffix == "году" {
            numbers::PREP
        } else if suffix == "года" {
            numbers::GEN
        } else {
            numbers::NOM
        };
        format!("{} {}", year_ordinal(year, case), suffix)
    })
}

fn expand_units(text: &str) -> String {
    // `sorted(UNITS, key=len, reverse=True)` order: 3-char, then 2-char in
    // insertion order, then 1-char in insertion order.
    let order = [
        "руб", "коп", "км", "см", "мм", "кг", "мг", "мл", "га", "м", "г", "л",
    ];
    let unit_re = order.join("|");
    let pattern = format!(r"(?<!\d)(\d{{1,9}})\s*({unit_re})\b");
    sub(text, &pattern, |caps| {
        let caps = caps;
        let value: i64 = caps.get(1).unwrap().as_str().parse().unwrap();
        let unit = caps.get(2).unwrap().as_str();
        let (_, (one, few, many)) = UNITS.iter().find(|(k, _)| *k == unit).unwrap();
        let case = case_for(text, caps.get(0).unwrap().start(), false, None);
        let mut words = numbers::cardinal(value, case, numbers::M);
        words.push(numbers::agree(value, one, few, many));
        words.join(" ")
    })
}

fn expand_decimals(text: &str) -> String {
    sub(text, r"(?<!\d)(\d{1,3},\d{1,3})(?!\d)", |caps| {
        let caps = caps;
        let raw = caps.get(0).unwrap().as_str();
        let gender = gender_of(&word_after(text, caps.get(0).unwrap().end()));
        let case = case_for(text, caps.get(0).unwrap().start(), false, None);
        numbers::decimal(raw, case, gender).join(" ")
    })
}

fn digit_by_digit(value: &str) -> Vec<String> {
    value
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| DIGIT_WORDS[c.to_digit(10).unwrap() as usize].to_string())
        .collect()
}

fn expand_phones(text: &str) -> String {
    sub(
        text,
        r"(номер[а-яё]*|телефон[а-яё]*|тел)\s+(\d[\d\s-]*\d)",
        |caps| {
            let caps = caps;
            let prefix = caps.get(1).unwrap().as_str();
            let digits: String = caps
                .get(2)
                .unwrap()
                .as_str()
                .chars()
                .filter(|c| c.is_ascii_digit())
                .collect();
            format!("{} {}", prefix, digit_by_digit(&digits).join(" "))
        },
    )
}

fn expand_long_numbers(text: &str) -> String {
    let context = rx(r"(?:инн|огрн|снилс|номер|телефон|тел|счёт|счет|билет|платформа)\s*$");
    sub(text, r"(?<!\d)\d+(?!\d)", |caps| {
        let caps = caps;
        let start = caps.get(0).unwrap().start();
        let end = caps.get(0).unwrap().end();
        let value = caps.get(0).unwrap().as_str();
        let before = text[..start].to_lowercase();
        let tail: String = before
            .chars()
            .rev()
            .take(40)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        if value.len() >= 8 || context.is_match(&tail).unwrap_or(false) {
            return digit_by_digit(value).join(" ");
        }
        let case = case_for(text, start, true, Some(end));
        let gender = gender_of(&word_after(text, end));
        numbers::cardinal(value.parse().unwrap_or(0), case, gender).join(" ")
    })
}

fn expand_latin(text: &str) -> String {
    sub(text, r"[a-z]+", |caps| {
        caps.get(0)
            .unwrap()
            .as_str()
            .chars()
            .map(|c| {
                LATIN_NAMES
                    .iter()
                    .find(|(l, _)| *l == c)
                    .map(|(_, n)| *n)
                    .unwrap_or(" ")
            })
            .collect::<Vec<_>>()
            .join(" ")
    })
}

fn punctuation(text: &str) -> String {
    let mut out = sub(text, r"\.{2,}", |_| "…".into());
    out = out
        .replace('„', "«")
        .replace('“', "«")
        .replace('”', "»")
        .replace('–', "—")
        .replace('−', "-")
        .replace('’', "'")
        .replace('`', "'");
    let mut open = true;
    out = sub(&out, r#"""#, |_| {
        open = !open;
        if open {
            "»".into()
        } else {
            "«".into()
        }
    });
    let allowed = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя ,.!?;:\"'()—…«»-–№%[]{}";
    out.chars()
        .map(|c| if allowed.contains(c) { c } else { ' ' })
        .collect()
}

fn spaces(text: &str) -> String {
    let mut out = sub(text, r"\s+", |_| " ".into());
    out = sub(&out, r"\s+([,.;:!?…»)\]])", |caps| {
        caps.get(1).unwrap().as_str().to_string()
    });
    out = sub(&out, r"([«(\[])\s+", |caps| {
        caps.get(1).unwrap().as_str().to_string()
    });
    out.trim().to_string()
}

/// Raw text -> lowercase words and inventory punctuation.
pub fn normalize(text: &str) -> String {
    let mut out = clean(text);
    out = expand_phones(&out);
    out = merge_thousands(&out);
    out = expand_scales(&out);
    out = expand_abbreviations(&out);
    out = expand_dates(&out);
    out = expand_time(&out);
    out = expand_years(&out);
    out = expand_money(&out);
    out = expand_percent(&out);
    out = expand_ordinals(&out);
    out = expand_units(&out);
    out = expand_decimals(&out);
    out = expand_long_numbers(&out);
    out = expand_latin(&out);
    out = punctuation(&out);
    spaces(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinals() {
        let cases = [
            ("0", "ноль"),
            ("21", "двадцать один"),
            ("40", "сорок"),
            ("100", "сто"),
            ("101", "сто один"),
            ("243", "двести сорок три"),
            ("1000", "тысяча"),
            ("2000", "две тысячи"),
            ("5000", "пять тысяч"),
            ("21000", "двадцать одна тысяча"),
            ("1000000", "миллион"),
        ];
        for (src, want) in cases {
            assert_eq!(normalize(src), want, "{src}");
        }
    }

    #[test]
    fn cases_from_prepositions() {
        let cases = [
            ("до 5", "до пяти"),
            ("от 3", "от трёх"),
            ("к 5", "к пяти"),
            ("по 2", "по двум"),
            ("без 5 минут", "без пяти минут"),
            ("в 5 часов", "в пять часов"),
            ("в 20 километрах", "в двадцати километрах"),
            ("на 5 пунктов", "на пять пунктов"),
            ("с 5 рублями", "с пятью рублями"),
        ];
        for (src, want) in cases {
            assert_eq!(normalize(src), want, "{src}");
        }
    }

    #[test]
    fn ordinals_dates_times() {
        let cases = [
            ("5-й", "пятый"),
            ("1-го", "первого"),
            ("20-е", "двадцатое"),
            ("2026-й", "две тысячи двадцать шестой"),
            ("40-й", "сороковой"),
            (
                "19 ноября 2026",
                "девятнадцатого ноября две тысячи двадцать шестого года",
            ),
            (
                "19.11.2026",
                "девятнадцатого ноября две тысячи двадцать шестого года",
            ),
            ("15:30", "пятнадцать тридцать"),
            ("09:05", "девять ноль пять"),
            ("В 2026 году", "в две тысячи двадцать шестом году"),
        ];
        for (src, want) in cases {
            assert_eq!(normalize(src), want, "{src}");
        }
    }

    #[test]
    fn money_units_abbreviations() {
        let cases = [
            ("1 500 ₽", "тысяча пятьсот рублей"),
            ("2 руб.", "два рубля"),
            ("5 коп.", "пять копеек"),
            ("5 км", "пять километров"),
            ("2 км", "два километра"),
            ("20%", "двадцать процентов"),
            ("3 кг", "три килограмма"),
            ("т. е.", "то есть"),
            ("ул. Пушкина", "улица пушкина"),
            ("5 тыс.", "пять тысяч"),
            ("№ 7", "номер семь"),
        ];
        for (src, want) in cases {
            assert_eq!(normalize(src), want, "{src}");
        }
    }

    #[test]
    fn latin_punctuation_yo() {
        let cases = [
            ("abc", "эй би си"),
            ("ok", "о кей"),
            ("Привет...", "привет…"),
            ("«Ёлка» — да!", "«ёлка» — да!"),
            ("“Ёлка”", "«ёлка»"),
            ("цена – 5", "цена — пять"),
            ("он сказал \"да\"", "он сказал «да»"),
            ("ЁЖ И МЁД", "ёж и мёд"),
        ];
        for (src, want) in cases {
            assert_eq!(normalize(src), want, "{src}");
        }
    }

    #[test]
    fn no_digits_or_latin_and_idempotent() {
        for text in [
            "Сегодня 23 января 2026 года мы встретимся в 15:30.",
            "Это стоит 1 500 ₽, скидка 20% до 30 июня.",
            "iPhone 15 и 2 кг яблок.",
        ] {
            let once = normalize(text);
            assert!(!once.contains(|c: char| c.is_ascii_digit() || c.is_ascii_alphabetic()));
            assert_eq!(normalize(&once), once);
        }
    }
}

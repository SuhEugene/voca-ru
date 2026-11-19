//! Russian cardinal and ordinal numerals with case and gender agreement.
//!
//! Direct port of `training/voca_ru/text/numbers.py`; behaviour must stay
//! identical (the parity harness compares normalised text).

pub type Case = &'static str;
pub type Gender = &'static str;

pub const NOM: Case = "nom";
pub const GEN: Case = "gen";
pub const DAT: Case = "dat";
pub const ACC: Case = "acc";
pub const INS: Case = "ins";
pub const PREP: Case = "prep";
pub const CASES: [Case; 6] = [NOM, GEN, DAT, ACC, INS, PREP];
pub const M: Gender = "m";
pub const F: Gender = "f";
pub const N: Gender = "n";

fn cidx(case: Case) -> usize {
    CASES.iter().position(|c| *c == case).unwrap_or(0)
}

const ZERO: [&str; 6] = ["ноль", "ноля", "нолю", "ноль", "нолём", "ноле"];

const ONE_M: [&str; 6] = ["один", "одного", "одному", "один", "одним", "одном"];
const ONE_F: [&str; 6] = ["одна", "одной", "одной", "одну", "одной", "одной"];
const ONE_N: [&str; 6] = ["одно", "одного", "одному", "одно", "одним", "одном"];
const TWO_M: [&str; 6] = ["два", "двух", "двум", "два", "двумя", "двух"];
const TWO_F: [&str; 6] = ["две", "двух", "двум", "две", "двумя", "двух"];

const THREE: [&str; 6] = ["три", "трёх", "трём", "три", "тремя", "трёх"];
const FOUR: [&str; 6] = [
    "четыре",
    "четырёх",
    "четырём",
    "четыре",
    "четырьмя",
    "четырёх",
];

/// (nom, gen/dat/prep, ins); acc = nom.
const TEENS: [(i64, [&str; 3]); 20] = [
    (5, ["пять", "пяти", "пятью"]),
    (6, ["шесть", "шести", "шестью"]),
    (7, ["семь", "семи", "семью"]),
    (8, ["восемь", "восьми", "восемью"]),
    (9, ["девять", "девяти", "девятью"]),
    (10, ["десять", "десяти", "десятью"]),
    (11, ["одиннадцать", "одиннадцати", "одиннадцатью"]),
    (12, ["двенадцать", "двенадцати", "двенадцатью"]),
    (13, ["тринадцать", "тринадцати", "тринадцатью"]),
    (14, ["четырнадцать", "четырнадцати", "четырнадцатью"]),
    (15, ["пятнадцать", "пятнадцати", "пятнадцатью"]),
    (16, ["шестнадцать", "шестнадцати", "шестнадцатью"]),
    (17, ["семнадцать", "семнадцати", "семнадцатью"]),
    (18, ["восемнадцать", "восемнадцати", "восемнадцатью"]),
    (19, ["девятнадцать", "девятнадцати", "девятнадцатью"]),
    (20, ["двадцать", "двадцати", "двадцатью"]),
    (30, ["тридцать", "тридцати", "тридцатью"]),
    (40, ["сорок", "сорока", "сорока"]),
    (50, ["пятьдесят", "пятидесяти", "пятьюдесятью"]),
    (60, ["шестьдесят", "шестидесяти", "шестьюдесятью"]),
];

const TENS_ODD: [(i64, [&str; 3]); 3] = [
    (70, ["семьдесят", "семидесяти", "семьюдесятью"]),
    (80, ["восемьдесят", "восьмидесяти", "восемьюдесятью"]),
    (90, ["девяносто", "девяноста", "девяноста"]),
];

const HUNDREDS: [(i64, [&str; 6]); 9] = [
    (100, ["сто", "ста", "ста", "сто", "ста", "ста"]),
    (
        200,
        [
            "двести",
            "двухсот",
            "двумстам",
            "двести",
            "двумястами",
            "двухстах",
        ],
    ),
    (
        300,
        [
            "триста",
            "трёхсот",
            "трёмстам",
            "триста",
            "тремястами",
            "трёхстах",
        ],
    ),
    (
        400,
        [
            "четыреста",
            "четырёхсот",
            "четырёмстам",
            "четыреста",
            "четырьмястами",
            "четырёхстах",
        ],
    ),
    (
        500,
        [
            "пятьсот",
            "пятисот",
            "пятистам",
            "пятьсот",
            "пятьюстами",
            "пятистах",
        ],
    ),
    (
        600,
        [
            "шестьсот",
            "шестисот",
            "шестистам",
            "шестьсот",
            "шестьюстами",
            "шестистах",
        ],
    ),
    (
        700,
        [
            "семьсот",
            "семисот",
            "семистам",
            "семьсот",
            "семьюстами",
            "семистах",
        ],
    ),
    (
        800,
        [
            "восемьсот",
            "восьмисот",
            "восьмистам",
            "восемьсот",
            "восемьюстами",
            "восьмистах",
        ],
    ),
    (
        900,
        [
            "девятьсот",
            "девятисот",
            "девятистам",
            "девятьсот",
            "девятьюстами",
            "девятистах",
        ],
    ),
];

struct Scale {
    value: i64,
    singular: [&'static str; 6],
    plural: [&'static str; 6],
    gender: Gender,
}

const SCALES: [Scale; 3] = [
    Scale {
        value: 1_000_000_000,
        singular: [
            "миллиард",
            "миллиарда",
            "миллиарду",
            "миллиард",
            "миллиардом",
            "миллиарде",
        ],
        plural: [
            "миллиарды",
            "миллиардов",
            "миллиардам",
            "миллиарды",
            "миллиардами",
            "миллиардах",
        ],
        gender: M,
    },
    Scale {
        value: 1_000_000,
        singular: [
            "миллион",
            "миллиона",
            "миллиону",
            "миллион",
            "миллионом",
            "миллионе",
        ],
        plural: [
            "миллионы",
            "миллионов",
            "миллионам",
            "миллионы",
            "миллионами",
            "миллионах",
        ],
        gender: M,
    },
    Scale {
        value: 1000,
        singular: ["тысяча", "тысячи", "тысяче", "тысячу", "тысячей", "тысяче"],
        plural: [
            "тысячи",
            "тысяч",
            "тысячам",
            "тысячи",
            "тысячами",
            "тысячах",
        ],
        gender: F,
    },
];

fn small(n: i64, case: Case, gender: Gender) -> Vec<String> {
    if n == 0 {
        return vec![ZERO[cidx(case)].to_string()];
    }
    if n == 1 {
        let table = match gender {
            F => ONE_F,
            N => ONE_N,
            _ => ONE_M,
        };
        return vec![table[cidx(case)].to_string()];
    }
    if n == 2 {
        let table = if gender == F { TWO_F } else { TWO_M };
        return vec![table[cidx(case)].to_string()];
    }
    if n == 3 {
        return vec![THREE[cidx(case)].to_string()];
    }
    if n == 4 {
        return vec![FOUR[cidx(case)].to_string()];
    }
    if let Some((_, forms)) = TEENS.iter().find(|(v, _)| *v == n) {
        return vec![forms[oblique_index(case)].to_string()];
    }
    if let Some((_, forms)) = TENS_ODD.iter().find(|(v, _)| *v == n) {
        return vec![forms[oblique_index(case)].to_string()];
    }
    if n > 20 {
        let tens = n / 10 * 10;
        let units = n % 10;
        let forms = TEENS
            .iter()
            .find(|(v, _)| *v == tens)
            .map(|(_, f)| f)
            .or_else(|| TENS_ODD.iter().find(|(v, _)| *v == tens).map(|(_, f)| f));
        if let Some(forms) = forms {
            let mut out = vec![forms[oblique_index(case)].to_string()];
            out.extend(small(units, case, gender));
            return out;
        }
    }
    Vec::new()
}

/// 0 = nom/acc, 1 = oblique (gen/dat/prep), 2 = instrumental.
fn oblique_index(case: Case) -> usize {
    if case == NOM || case == ACC {
        0
    } else if case == INS {
        2
    } else {
        1
    }
}

fn under_thousand(n: i64, case: Case, gender: Gender) -> Vec<String> {
    if n < 100 {
        return small(n, case, gender);
    }
    let hundreds = n / 100;
    let rest = n % 100;
    let mut out = vec![hundred_form(hundreds * 100, case).to_string()];
    if rest != 0 {
        out.extend(small(rest, case, gender));
    }
    out
}

fn hundred_form(n: i64, case: Case) -> &'static str {
    HUNDREDS
        .iter()
        .find(|(v, _)| *v == n)
        .map(|(_, f)| f[cidx(case)])
        .unwrap_or("")
}

fn scale_form(
    count: i64,
    singular: &[&'static str; 6],
    plural: &[&'static str; 6],
    case: Case,
) -> &'static str {
    let mod100 = count % 100;
    let mod10 = count % 10;
    let oblique = case != NOM && case != ACC;
    if mod10 == 1 && mod100 != 11 {
        return singular[cidx(case)];
    }
    if oblique {
        return plural[cidx(case)];
    }
    if (2..=4).contains(&mod10) && !(11..=14).contains(&mod100) {
        return singular[cidx(GEN)];
    }
    plural[cidx(GEN)]
}

pub fn cardinal(n: i64, case: Case, gender: Gender) -> Vec<String> {
    if n < 0 {
        let mut out = vec!["минус".to_string()];
        out.extend(cardinal(-n, case, gender));
        return out;
    }
    if n < 1000 {
        return under_thousand(n, case, gender);
    }
    let mut out: Vec<String> = Vec::new();
    let mut rest = n;
    for scale in SCALES.iter() {
        let count = rest / scale.value;
        rest %= scale.value;
        if count != 0 {
            if count == 1 {
                out.push(scale.singular[cidx(case)].to_string());
            } else {
                out.extend(cardinal(count, case, scale.gender));
                out.push(scale_form(count, &scale.singular, &scale.plural, case).to_string());
            }
        }
    }
    if rest != 0 {
        out.extend(under_thousand(rest, case, gender));
    }
    out
}

pub fn agree(n: i64, one: &str, few: &str, many: &str) -> String {
    let mod100 = n % 100;
    let mod10 = n % 10;
    if (11..=14).contains(&mod100) {
        many.to_string()
    } else if mod10 == 1 {
        one.to_string()
    } else if (2..=4).contains(&mod10) {
        few.to_string()
    } else {
        many.to_string()
    }
}

const ORD_STEMS: [(i64, &str); 36] = [
    (0, "нулев"),
    (1, "перв"),
    (2, "втор"),
    (3, "трет"),
    (4, "четвёрт"),
    (5, "пят"),
    (6, "шест"),
    (7, "седьм"),
    (8, "восьм"),
    (9, "девят"),
    (10, "десят"),
    (11, "одиннадцат"),
    (12, "двенадцат"),
    (13, "тринадцат"),
    (14, "четырнадцат"),
    (15, "пятнадцат"),
    (16, "шестнадцат"),
    (17, "семнадцат"),
    (18, "восемнадцат"),
    (19, "девятнадцат"),
    (20, "двадцат"),
    (30, "тридцат"),
    (40, "сороков"),
    (50, "пятидесят"),
    (60, "шестидесят"),
    (70, "семидесят"),
    (80, "восьмидесят"),
    (90, "девяност"),
    (100, "сот"),
    (200, "двухсот"),
    (300, "трёхсот"),
    (400, "четырёхсот"),
    (500, "пятисот"),
    (600, "шестисот"),
    (700, "семисот"),
    (800, "восьмисот"),
];

const ORD_STEMS_HIGH: [(i64, &str); 7] = [
    (900, "девятисот"),
    (1000, "тысячн"),
    (2000, "двухтысячн"),
    (3000, "трёхтысячн"),
    (4000, "четырёхтысячн"),
    (5000, "пятитысячн"),
    (1_000_000, "миллионн"),
];

fn stem(n: i64) -> Option<&'static str> {
    ORD_STEMS
        .iter()
        .chain(ORD_STEMS_HIGH.iter())
        .find(|(v, _)| *v == n)
        .map(|(_, s)| *s)
        .or(if n == 1_000_000_000 {
            Some("миллиардн")
        } else {
            None
        })
}

fn is_soft_stem(s: &str) -> bool {
    matches!(s, "нулев" | "втор" | "шест" | "седьм" | "восьм" | "сороков")
}

fn ord_word(stem: &str, case: Case, gender: Gender) -> String {
    let key = (gender, case);
    if is_soft_stem(stem) {
        let ending = match key {
            (M, NOM) => "ой",
            (M, GEN) => "ого",
            (M, DAT) => "ому",
            (M, ACC) => "ой",
            (M, INS) => "ым",
            (M, PREP) => "ом",
            (F, NOM) => "ая",
            (F, GEN) => "ой",
            (F, DAT) => "ой",
            (F, ACC) => "ую",
            (F, INS) => "ой",
            (F, PREP) => "ой",
            (N, NOM) => "ое",
            (N, GEN) => "ого",
            (N, DAT) => "ому",
            (N, ACC) => "ое",
            (N, INS) => "ым",
            (N, PREP) => "ом",
            _ => "",
        };
        return format!("{stem}{ending}");
    }
    if stem == "трет" {
        return match key {
            (M, NOM) => "третий",
            (M, GEN) => "третьего",
            (M, DAT) => "третьему",
            (M, ACC) => "третий",
            (M, INS) => "третьим",
            (M, PREP) => "третьем",
            (F, NOM) => "третья",
            (F, GEN) => "третьей",
            (F, DAT) => "третьей",
            (F, ACC) => "третью",
            (F, INS) => "третьей",
            (F, PREP) => "третьей",
            (N, NOM) => "третье",
            (N, GEN) => "третьего",
            (N, DAT) => "третьему",
            (N, ACC) => "третье",
            (N, INS) => "третьим",
            (N, PREP) => "третьем",
            _ => "третий",
        }
        .to_string();
    }
    let endings: [&str; 6] = match gender {
        F => ["ая", "ой", "ой", "ую", "ой", "ой"],
        N => ["ое", "ого", "ому", "ое", "ым", "ом"],
        _ => ["ый", "ого", "ому", "ый", "ым", "ом"],
    };
    format!("{stem}{}", endings[cidx(case)])
}

pub fn ordinal(n: i64, case: Case, gender: Gender) -> String {
    if n <= 0 {
        return ord_word("нулев", case, gender);
    }
    if let Some(s) = stem(n) {
        return ord_word(s, case, gender);
    }
    if (11..=19).contains(&n) {
        return ord_word(stem(n).unwrap(), case, gender);
    }
    if n < 100 {
        let tens = n / 10;
        let units = n % 10;
        if units == 0 {
            return ord_word(stem(tens * 10).unwrap(), case, gender);
        }
        return format!(
            "{} {}",
            cardinal(tens * 10, NOM, M).join(" "),
            ord_word(stem(units).unwrap(), case, gender)
        );
    }
    if n < 1000 {
        let hundreds = n / 100;
        let rest = n % 100;
        if rest == 0 {
            return ord_word(stem(hundreds * 100).unwrap(), case, gender);
        }
        return format!(
            "{} {}",
            cardinal(hundreds * 100, NOM, M).join(" "),
            ordinal(rest, case, gender)
        );
    }
    for (value, _name) in [
        (1_000_000_000i64, "миллиард"),
        (1_000_000, "миллион"),
        (1000, "тысяч"),
    ] {
        if n >= value {
            let count = n / value;
            let rest = n % value;
            if rest == 0 {
                if let Some(s) = stem(n) {
                    return ord_word(s, case, gender);
                }
                if count == 1 {
                    return ord_word(stem(value).unwrap_or("тысячн"), case, gender);
                }
                return format!(
                    "{} {}",
                    cardinal(n, NOM, M).join(" "),
                    ord_word(stem(value).unwrap_or("тысячн"), case, gender)
                );
            }
            let head = cardinal(n - rest, NOM, M).join(" ");
            return format!("{head} {}", ordinal(rest, case, gender));
        }
    }
    cardinal(n, case, gender).join(" ")
}

pub fn decimal(value: &str, case: Case, gender: Gender) -> Vec<String> {
    let (integer_part, fraction_part) = value.split_once(',').unwrap_or((value, ""));
    let integer: i64 = integer_part.parse().unwrap_or(0);
    let fraction = fraction_part.trim_end_matches('0');
    if fraction.is_empty() {
        return cardinal(integer, case, gender);
    }
    if integer == 1 && fraction == "5" {
        return vec![if gender == M || gender == N {
            "полтора"
        } else {
            "полторы"
        }
        .to_string()];
    }
    let mut out = cardinal(integer, case, gender);
    let frac_value: i64 = fraction.parse().unwrap_or(0);
    let frac_den = 10i64.pow(fraction.len() as u32);
    let den_words: [&str; 3] = match frac_den {
        100 => ["сотая", "сотых", "сотых"],
        1000 => ["тысячная", "тысячных", "тысячных"],
        _ => ["десятая", "десятых", "десятых"],
    };
    if integer != 0 {
        let word = if integer % 10 == 1 && integer % 100 != 11 {
            match gender {
                F => "целая",
                N => "целое",
                _ => "целый",
            }
        } else if (2..=4).contains(&(integer % 10)) && !(11..=14).contains(&(integer % 100)) {
            "целых"
        } else {
            "целых"
        };
        out.push(word.to_string());
    }
    out.extend(cardinal(frac_value, NOM, F));
    let den = if frac_value % 10 == 1 && frac_value % 100 != 11 {
        den_words[0]
    } else if (2..=4).contains(&(frac_value % 10)) && !(11..=14).contains(&(frac_value % 100)) {
        den_words[1]
    } else {
        den_words[2]
    };
    out.push(den.to_string());
    out
}

//! Numbers as words.

use crate::pyunicode as py;

pub const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 8] = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES: [&str; 6] = ["thousand", "million", "billion", "trillion", "quadrillion", "quintillion"];

/// `1234` -> `"one thousand two hundred thirty four"`.
pub fn cardinal(number: u128) -> String {
    if number < 20 {
        return ONES[number as usize].to_string();
    }
    if number < 100 {
        let (tens, ones) = (number / 10, number % 10);
        return if ones != 0 { format!("{} {}", TENS[tens as usize - 2], ONES[ones as usize]) } else { TENS[tens as usize - 2].to_string() };
    }
    if number < 1000 {
        let (hundreds, rest) = (number / 100, number % 100);
        return if rest != 0 { format!("{} hundred {}", ONES[hundreds as usize], cardinal(rest)) } else { format!("{} hundred", ONES[hundreds as usize]) };
    }
    if number >= 1000u128.pow(SCALES.len() as u32 + 1) {
        return digits(&number.to_string(), "zero");
    }
    let mut parts = Vec::new();
    let mut number = number;
    for power in (1..=SCALES.len() as u32).rev() {
        let unit = 1000u128.pow(power);
        let count = number / unit;
        number %= unit;
        if count != 0 {
            parts.push(format!("{} {}", cardinal(count), SCALES[power as usize - 1]));
        }
    }
    if number != 0 {
        parts.push(cardinal(number));
    }
    parts.join(" ")
}

/// A run of digits of any length as a number: `(value if it fits, digits without leading zeros)`.
fn parse(text: &str) -> (Option<u128>, String) {
    let ascii = py::ascii_digits(text);
    let trimmed = ascii.trim_start_matches('0');
    let trimmed = if trimmed.is_empty() { "0" } else { trimmed };
    (if trimmed.len() <= 38 { trimmed.parse().ok() } else { None }, trimmed.to_string())
}

/// `cardinal(int(text))` for a run of digits of any length.
pub fn cardinal_of(text: &str) -> String {
    match parse(text) {
        (Some(value), _) => cardinal(value),
        (None, plain) => digits(&plain, "zero"),
    }
}

/// `int(text)` when it is small enough to matter in a comparison.
pub fn value_of(text: &str) -> Option<u128> {
    parse(text).0
}

fn ordinal_words(cardinal: String) -> String {
    let mut words: Vec<String> = cardinal.split(' ').map(String::from).collect();
    let last = words.pop().unwrap();
    let said = match last.as_str() {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        _ if last.ends_with('y') => format!("{}ieth", &last[..last.len() - 1]),
        _ => format!("{last}th"),
    };
    words.push(said);
    words.join(" ")
}

pub fn ordinal(number: u128) -> String {
    ordinal_words(cardinal(number))
}

pub fn ordinal_of(text: &str) -> String {
    ordinal_words(cardinal_of(text))
}

/// The digits of `text` one by one.
pub fn digits(text: &str, zero: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for c in text.chars() {
        if py::is_digit(c) {
            out.push(if c == '0' { zero } else { ONES[py::digit_value(c).unwrap_or(0).min(9) as usize] });
        }
    }
    out.join(" ")
}

/// The way a year is said: 1984, 1905, 1900, 2005, 2026.
pub fn year(number: u128) -> String {
    if !(1000..=2999).contains(&number) {
        return cardinal(number);
    }
    let (high, low) = (number / 100, number % 100);
    if low == 0 {
        return if high % 10 == 0 { cardinal(number) } else { format!("{} hundred", cardinal(high)) };
    }
    if high == 20 && low < 10 {
        return format!("two thousand {}", ONES[low as usize]);
    }
    if low < 10 { format!("{} oh {}", cardinal(high), ONES[low as usize]) } else { format!("{} {}", cardinal(high), cardinal(low)) }
}

/// A number read two digits at a time, the way model numbers are.
pub fn pairs(number: u128) -> String {
    let (high, low) = (number / 100, number % 100);
    if !(100..=9999).contains(&number) || (low == 0 && (high < 10 || high % 10 == 0)) {
        return cardinal(number);
    }
    if low == 0 {
        return format!("{} hundred", cardinal(high));
    }
    if low < 10 { format!("{} oh {}", cardinal(high), ONES[low as usize]) } else { format!("{} {}", cardinal(high), cardinal(low)) }
}

/// An unsigned number as written: `1,234`, `3.14`, `007`.
pub fn number_words(text: &str) -> String {
    let plain = text.replace(',', "");
    let (whole, fraction) = match plain.split_once('.') {
        Some((whole, fraction)) => (whole, fraction),
        None => (plain.as_str(), ""),
    };
    if !fraction.is_empty() {
        return format!("{} point {}", if whole.is_empty() { String::new() } else { cardinal_of(whole) }, digits(fraction, "zero"));
    }
    let length = whole.chars().count();
    if (length > 1 && whole.starts_with('0')) || length > 18 {
        return digits(whole, "zero");
    }
    cardinal_of(whole)
}

/// `2:03:45` as a length of time.
pub fn duration(hours: u128, minutes: u128, seconds: u128) -> String {
    let parts: Vec<String> = [(hours, "hour"), (minutes, "minute"), (seconds, "second")]
        .iter()
        .filter(|(count, _)| *count != 0)
        .map(|(count, name)| format!("{} {}{}", cardinal(*count), name, if *count == 1 { "" } else { "s" }))
        .collect();
    match parts.len() {
        0 => "zero seconds".to_string(),
        1 => parts[0].clone(),
        n => format!("{} and {}", parts[..n - 1].join(", "), parts[n - 1]),
    }
}

/// Plural of a spelled number, for decades: "nineteen nineties".
pub fn plural_of(words: &str) -> String {
    let (head, last) = match words.rsplit_once(' ') {
        Some((head, last)) => (head, last),
        None => ("", words),
    };
    let last = if last.ends_with('y') { format!("{}ies", &last[..last.len() - 1]) } else { format!("{last}s") };
    if head.is_empty() { last } else { format!("{head} {last}") }
}

/// "three quarters", "one half"; after a whole number "a half", "an eighth".
pub fn fraction_words(top: u128, bottom: u128, article: bool) -> String {
    let (one, many) = match bottom {
        2 => ("half".to_string(), "halves".to_string()),
        4 => ("quarter".to_string(), "quarters".to_string()),
        _ => (ordinal(bottom), format!("{}s", ordinal(bottom))),
    };
    if top != 1 {
        return format!("{} {}", cardinal(top), many);
    }
    let lead = if article { if one.starts_with(['a', 'e', 'i', 'o', 'u']) { "an " } else { "a " } } else { "one " };
    format!("{lead}{one}")
}

/// One part of a version number: "4" four, "12" twelve, "03" zero three, "129" one two nine.
pub fn version_part(part: &str) -> String {
    if part.chars().count() > 2 { digits(part, "zero") } else { number_words(part) }
}

/// The part after the point of a version: "12" twelve, "04" oh four.
pub fn minor(part: &str) -> String {
    if part.starts_with('0') { digits(part, "oh") } else { cardinal_of(part) }
}

/// A number in a code that is said as a number: one or two digits, or a round one.
pub fn short_number(text: &str) -> bool {
    if text.starts_with('0') {
        return false;
    }
    let mut chars = text.chars();
    chars.next();
    let rest = chars.as_str();
    text.chars().count() <= 2 || (!rest.is_empty() && rest.chars().all(|c| c == '0'))
}

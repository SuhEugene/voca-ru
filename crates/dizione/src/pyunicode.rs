//! Character classes and string helpers with the semantics of the Python the
//! reference implementation runs on (tables generated from CPython).

use crate::unicode_tables as t;

fn find(table: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    table
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                std::cmp::Ordering::Greater
            } else if cp > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `str.isalnum()` for one character (what `\w` matches, apart from `_`).
pub fn is_alnum(c: char) -> bool {
    if c.is_ascii() { c.is_ascii_alphanumeric() } else { find(&t::ALNUM, c) }
}
pub fn is_alpha(c: char) -> bool {
    if c.is_ascii() { c.is_ascii_alphabetic() } else { find(&t::ALPHA, c) }
}
/// `str.isdigit()`.
pub fn is_digit(c: char) -> bool {
    if c.is_ascii() { c.is_ascii_digit() } else { find(&t::DIGIT, c) }
}
/// `str.isdecimal()` (what `\d` matches).
pub fn is_decimal(c: char) -> bool {
    if c.is_ascii() { c.is_ascii_digit() } else { find(&t::DECIMAL, c) }
}
/// `str.isspace()` (what `\s` matches).
pub fn is_space(c: char) -> bool {
    if c.is_ascii() {
        matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c' | '\x1c'..='\x1f')
    } else {
        find(&t::SPACE, c)
    }
}
/// Value of a digit character (`int(c)` for decimals).
pub fn digit_value(c: char) -> Option<u32> {
    if c.is_ascii() {
        return c.to_digit(10);
    }
    let cp = c as u32;
    let i = t::DIGIT_VALUES.partition_point(|&(start, _, _)| start <= cp);
    if i == 0 {
        return None;
    }
    let (start, value, len) = t::DIGIT_VALUES[i - 1];
    if cp < start + len as u32 { Some(value as u32 + (cp - start)) } else { None }
}

/// The digits of `text` as ASCII (any Unicode decimal digit accepted, as `int()` does).
pub fn ascii_digits(text: &str) -> String {
    text.chars().filter_map(|c| digit_value(c).and_then(|v| char::from_digit(v, 10))).collect()
}

/// `int(text)` for a short run of digits.
pub fn int(text: &str) -> u64 {
    let mut n: u64 = 0;
    for c in text.chars() {
        if let Some(v) = digit_value(c) {
            n = n.saturating_mul(10).saturating_add(v as u64);
        }
    }
    n
}

pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}
pub fn lstrip(s: &str) -> &str {
    s.trim_start_matches(is_space)
}
pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}
/// `str.split()` with no argument.
pub fn split(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_space).filter(|part| !part.is_empty())
}

/// `str.isupper()`: at least one cased character and no lower-case one.
pub fn is_upper(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_lowercase() {
            return false;
        }
        if c.is_uppercase() {
            cased = true;
        }
    }
    cased
}
/// `str.islower()`.
pub fn is_lower(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_uppercase() {
            return false;
        }
        if c.is_lowercase() {
            cased = true;
        }
    }
    cased
}
/// `str.isalpha()`.
pub fn all_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_alpha)
}
/// `str.isdigit()`.
pub fn all_digit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_digit)
}
pub fn first(s: &str) -> Option<char> {
    s.chars().next()
}
/// The last `count` characters of `s`.
pub fn tail(s: &str, count: usize) -> &str {
    match s.char_indices().rev().nth(count.saturating_sub(1)) {
        Some((i, _)) if count > 0 => &s[i..],
        _ if count == 0 => "",
        _ => s,
    }
}

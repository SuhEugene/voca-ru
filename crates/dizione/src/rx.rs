//! Regular expressions written in Python's dialect, compiled for `fancy-regex`.
//!
//! `\w`, `\d`, `\s` and `\b` mean what they mean in Python (the classes are
//! generated from CPython). Each pattern has two compiled forms: a narrow one
//! whose classes cover ASCII only, exact for text whose non-ASCII characters
//! are neither letters, digits nor spaces (nearly all text), and a wide one.

use std::sync::OnceLock;

use fancy_regex::{Captures, Regex};

use crate::pyunicode;
use crate::unicode_tables as t;

fn class(table: &[(u32, u32)]) -> String {
    let mut out = String::new();
    for &(lo, hi) in table {
        if lo == hi {
            out.push_str(&format!("\\x{{{lo:X}}}"));
        } else {
            out.push_str(&format!("\\x{{{lo:X}}}-\\x{{{hi:X}}}"));
        }
    }
    out
}

struct Classes {
    word: String,
    digit: String,
    space: String,
}

fn classes(wide: bool) -> &'static Classes {
    static NARROW: OnceLock<Classes> = OnceLock::new();
    static WIDE: OnceLock<Classes> = OnceLock::new();
    if wide {
        WIDE.get_or_init(|| Classes { word: class(&t::ALNUM) + "_", digit: class(&t::DECIMAL), space: class(&t::SPACE) })
    } else {
        NARROW.get_or_init(|| Classes {
            word: "A-Za-z0-9_".into(),
            digit: "0-9".into(),
            space: " \\t\\n\\r\\x0b\\x0c\\x1c-\\x1f".into(),
        })
    }
}

/// Python pattern -> pattern for `fancy-regex`.
pub fn translate(pattern: &str, wide: bool) -> String {
    let c = classes(wide);
    let mut out = String::with_capacity(pattern.len() + 64);
    let mut chars = pattern.chars().peekable();
    let mut in_class = false;
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                let next = chars.next().expect("pattern ends with a backslash");
                match (next, in_class) {
                    ('w', true) => out.push_str(&c.word),
                    ('w', false) => { out.push('['); out.push_str(&c.word); out.push(']'); }
                    ('d', true) => out.push_str(&c.digit),
                    ('d', false) => { out.push('['); out.push_str(&c.digit); out.push(']'); }
                    ('s', true) => out.push_str(&c.space),
                    ('s', false) => { out.push('['); out.push_str(&c.space); out.push(']'); }
                    ('S', false) => { out.push_str("[^"); out.push_str(&c.space); out.push(']'); }
                    ('b', false) => {
                        let w = &c.word;
                        out.push_str(&format!("(?:(?<=[{w}])(?![{w}])|(?<![{w}])(?=[{w}]))"));
                    }
                    ('S', true) | ('b', true) | ('W', _) | ('D', _) | ('B', _) => panic!("unsupported escape in {pattern}"),
                    _ => { out.push('\\'); out.push(next); }
                }
            }
            '[' if !in_class => { in_class = true; out.push('['); }
            ']' if in_class => { in_class = false; out.push(']'); }
            '$' if !in_class => out.push_str("(?=\\n?\\z)"),
            _ => out.push(ch),
        }
    }
    out
}

/// True when `text` needs the wide form of the patterns.
pub fn is_wide(text: &str) -> bool {
    !text.is_ascii() && text.chars().any(|c| !c.is_ascii() && (pyunicode::is_alnum(c) || pyunicode::is_space(c)))
}

pub struct Rx {
    narrow: OnceLock<Regex>,
    wide: OnceLock<Regex>,
}

impl Rx {
    pub const fn new() -> Self {
        Rx { narrow: OnceLock::new(), wide: OnceLock::new() }
    }
    pub fn get(&self, wide: bool, pattern: impl FnOnce() -> String) -> &Regex {
        let slot = if wide { &self.wide } else { &self.narrow };
        slot.get_or_init(|| {
            let source = pattern();
            let translated = translate(&source, wide);
            Regex::new(&translated).unwrap_or_else(|error| panic!("pattern {source}: {error}"))
        })
    }
}

/// `rx!(text, "pattern")` or `rx!(text, expression giving a String)`: the
/// compiled pattern in the form `text` needs.
macro_rules! rx {
    ($text:expr, $pattern:expr) => {{
        static CELL: $crate::rx::Rx = $crate::rx::Rx::new();
        CELL.get($crate::rx::is_wide($text), || String::from($pattern))
    }};
}
pub(crate) use rx;

pub fn group<'t>(caps: &Captures<'t>, index: usize) -> Option<&'t str> {
    caps.get(index).map(|m| m.as_str())
}
/// Group as text, `""` when it did not take part.
pub fn text<'t>(caps: &Captures<'t>, index: usize) -> &'t str {
    caps.get(index).map_or("", |m| m.as_str())
}
pub fn named<'t>(caps: &Captures<'t>, name: &str) -> Option<&'t str> {
    caps.name(name).map(|m| m.as_str())
}
pub fn start(caps: &Captures) -> usize {
    caps.get(0).unwrap().start()
}
pub fn end(caps: &Captures) -> usize {
    caps.get(0).unwrap().end()
}

/// `re.sub(pattern, function, text)`.
pub fn sub(re: &Regex, text: &str, mut replace: impl FnMut(&Captures) -> String) -> String {
    let mut out = String::with_capacity(text.len() + 16);
    let mut position = 0;
    let mut copied = 0;
    while position <= text.len() {
        let caps = match re.captures_from_pos(text, position) {
            Ok(Some(caps)) => caps,
            _ => break,
        };
        let whole = caps.get(0).unwrap();
        out.push_str(&text[copied..whole.start()]);
        out.push_str(&replace(&caps));
        copied = whole.end();
        if whole.end() == whole.start() {
            match text[whole.end()..].chars().next() {
                Some(c) => position = whole.end() + c.len_utf8(),
                None => break,
            }
        } else {
            position = whole.end();
        }
    }
    out.push_str(&text[copied..]);
    out
}

pub fn found(re: &Regex, text: &str) -> bool {
    re.is_match(text).unwrap_or(false)
}
/// `re.findall` for a pattern without groups.
pub fn find_all<'t>(re: &Regex, text: &'t str) -> Vec<&'t str> {
    re.find_iter(text).filter_map(|m| m.ok()).map(|m| m.as_str()).collect()
}

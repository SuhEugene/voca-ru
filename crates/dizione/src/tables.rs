//! Hand-written tables that ship with the source (`data/*.txt`), embedded.
//!
//! One entry per line, columns separated by tabs or by two or more spaces,
//! `#` starts a comment line. Pronunciations are phones of the inventory with
//! stress marks or CMUdict notation; the reader tells them apart.

use std::collections::HashMap;

use crate::convert::{from_text, is_arpabet};
use crate::pyunicode as py;
use crate::symbols::Phones;

pub const ABBREVIATIONS: &str = include_str!("../data/abbreviations.txt");
pub const CHAT: &str = include_str!("../data/chat.txt");
pub const ENDINGS: &str = include_str!("../data/endings.txt");
pub const EXCEPTIONS: &str = include_str!("../data/exceptions.txt");
pub const FUNCTION_WORDS: &str = include_str!("../data/function_words.txt");
pub const HOMOGRAPH_RULES: &str = include_str!("../data/homograph_rules.txt");
pub const LETTERS: &str = include_str!("../data/letters.txt");
pub const PHRASES: &str = include_str!("../data/phrases.txt");
pub const UNITS: &str = include_str!("../data/units.txt");

const STRESS_CLASSES: [&str; 8] = ["u", "uf", "ur", "s", "ss", "si", "pf", "p"];

/// `re.split(r"\t+| {2,}", line)`.
fn columns(line: &str) -> Vec<&str> {
    let bytes = line.as_bytes();
    let mut out = Vec::new();
    let (mut start, mut i) = (0, 0);
    while i < bytes.len() {
        let mut j = i;
        if bytes[i] == b'\t' {
            while j < bytes.len() && bytes[j] == b'\t' { j += 1; }
        } else if bytes[i] == b' ' && i + 1 < bytes.len() && bytes[i + 1] == b' ' {
            while j < bytes.len() && bytes[j] == b' ' { j += 1; }
        }
        if j > i {
            out.push(py::strip(&line[start..i]));
            start = j;
            i = j;
        } else {
            i += 1;
        }
    }
    out.push(py::strip(&line[start..]));
    out
}

/// Rows of a table: `(line number, columns)`.
pub fn rows(text: &'static str) -> Vec<(usize, Vec<&'static str>)> {
    let mut out = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = py::strip(line);
        if !line.is_empty() && !line.starts_with('#') {
            out.push((number + 1, columns(line)));
        }
    }
    out
}

pub type FunctionWord = (&'static str, Phones, Option<Phones>);

/// word -> (class, form, final form).
pub fn function_words() -> Result<HashMap<&'static str, FunctionWord>, String> {
    let mut table = HashMap::new();
    for (number, c) in rows(FUNCTION_WORDS) {
        if !(c.len() == 3 || c.len() == 4) || !STRESS_CLASSES.contains(&c[1]) {
            return Err(format!("function_words.txt line {number}: expected word, class, form[, final form]"));
        }
        let last = if c.len() == 4 { Some(from_text(c[3])?) } else { None };
        table.insert(c[0], (c[1], from_text(c[2])?, last));
    }
    Ok(table)
}

pub type Phrase = (&'static str, Vec<Phones>, Option<Phones>);

/// (word, word, ...) -> (class, forms, final form).
pub fn phrases() -> Result<HashMap<Vec<&'static str>, Phrase>, String> {
    let mut table = HashMap::new();
    for (number, c) in rows(PHRASES) {
        if !(c.len() == 3 || c.len() == 4) || !STRESS_CLASSES.contains(&c[1]) {
            return Err(format!("phrases.txt line {number}: expected words, class, form[, final form]"));
        }
        let forms = c[2].split('|').map(from_text).collect::<Result<Vec<_>, _>>()?;
        let key: Vec<&'static str> = py::split(c[0]).collect();
        if forms.len() > 1 && (c[1] != "p" || forms.len() != key.len()) {
            return Err(format!("phrases.txt line {number}: a form with boundaries needs class p and one part per word"));
        }
        let last = if c.len() == 4 { Some(from_text(c[3])?) } else { None };
        table.insert(key, (c[1], forms, last));
    }
    Ok(table)
}

#[derive(Clone, Debug)]
pub enum Pronunciation {
    Arpabet(Vec<&'static str>),
    Phones(Phones),
}

fn pronunciation(text: &'static str) -> Result<Pronunciation, String> {
    let symbols: Vec<&'static str> = py::split(text).collect();
    if is_arpabet(&symbols) { Ok(Pronunciation::Arpabet(symbols)) } else { Ok(Pronunciation::Phones(from_text(text)?)) }
}

/// word -> pronunciation, for the built-in exception list.
pub fn pronunciations(text: &'static str) -> Result<HashMap<String, Pronunciation>, String> {
    let mut table = HashMap::new();
    for (number, c) in rows(text) {
        if c.len() != 2 {
            return Err(format!("{number}: expected a word and its pronunciation, separated by a tab or two spaces"));
        }
        if py::split(c[0]).count() == 1 {
            table.insert(c[0].to_lowercase(), pronunciation(c[1])?);
        }
    }
    Ok(table)
}

/// (word, word, ...) -> one pronunciation per word, for names of several words.
pub fn names(text: &'static str) -> Result<HashMap<Vec<String>, Vec<Pronunciation>>, String> {
    let mut table = HashMap::new();
    for (number, c) in rows(text) {
        let lower = c[0].to_lowercase();
        let words: Vec<String> = py::split(&lower).map(String::from).collect();
        if c.len() != 2 || words.len() < 2 {
            continue;
        }
        let parts: Vec<&'static str> = c[1].split('|').map(py::strip).collect();
        if parts.len() != words.len() {
            return Err(format!("{number}: {:?} has {} words and {} pronunciations", c[0], words.len(), parts.len()));
        }
        table.insert(words, parts.into_iter().map(pronunciation).collect::<Result<Vec<_>, _>>()?);
    }
    Ok(table)
}

/// Two-column table as a dictionary (second column may be empty).
pub fn mapping(text: &'static str) -> HashMap<&'static str, &'static str> {
    rows(text).into_iter().map(|(_, c)| (c[0], if c.len() > 1 { c[1] } else { "" })).collect()
}

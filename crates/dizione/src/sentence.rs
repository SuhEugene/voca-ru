//! Sentence rules: what a word's pronunciation owes to its neighbours, within
//! one clause (sentence stress, "the"/"to" before a vowel, linking r, flaps).

use crate::symbols::{is_vowel, Ph, Phones};

fn split_rhotic(phone: &str) -> Option<&'static str> {
    Some(match phone {
        "ɑːɹ" => "ɑː",
        "ɔːɹ" => "ɔː",
        "oːɹ" => "oː",
        "ɪɹ" => "ɪ",
        "ɛɹ" => "ɛ",
        "ʊɹ" => "ʊ",
        _ => return None,
    })
}

#[derive(Clone, Debug)]
pub struct Unit {
    pub word: String,
    pub phones: Phones,
    /// Stress class of function_words.txt; "c" = stress as given.
    pub kind: &'static str,
    /// Form to use when the word is promoted.
    pub last: Option<Phones>,
    /// No word boundary before this unit.
    pub attach: bool,
}

impl Unit {
    fn last_word(&self) -> &str {
        self.word.rsplit(' ').next().unwrap()
    }
    fn first_word(&self) -> &str {
        self.word.split(' ').next().unwrap()
    }
}

fn unstress(phones: &[Ph]) -> Phones {
    phones.iter().map(|&(p, _)| (p, 0)).collect()
}

/// Give the word a primary stress: on its marked vowel, else on the first vowel.
pub fn promote(phones: &[Ph]) -> Phones {
    if phones.iter().any(|&(_, s)| s == 1) {
        return phones.to_vec();
    }
    let target = match phones.iter().rposition(|&(_, s)| s == 2) {
        Some(index) => Some(index),
        None => phones.iter().position(|&(p, _)| is_vowel(p)),
    };
    phones.iter().enumerate().map(|(i, &(p, s))| (p, if Some(i) == target { 1 } else { s })).collect()
}

/// Resolve the stress class of every unit of a clause, right to left.
pub fn stress_clause(units: &mut [Unit]) {
    let mut primary_follows = false;
    let count = units.len();
    for index in (0..count).rev() {
        let unit = &units[index];
        let promoted = || unit.last.clone().unwrap_or_else(|| promote(&unit.phones));
        let phones = match unit.kind {
            "u" => unstress(&unit.phones),
            "uf" => if index == count - 1 { promoted() } else { unstress(&unit.phones) },
            "ur" => if primary_follows { unstress(&unit.phones) } else { promoted() },
            "s" => if primary_follows { unit.phones.clone() } else { promoted() },
            "si" => {
                if !primary_follows {
                    promoted()
                } else if index == 0 {
                    unit.phones.clone()
                } else {
                    unstress(&unit.phones)
                }
            }
            "pf" => if index == count - 1 { promoted() } else { unit.phones.clone() },
            _ => unit.phones.clone(),
        };
        primary_follows = primary_follows || phones.iter().any(|&(_, s)| s == 1);
        units[index].phones = phones;
    }
    if !units.is_empty() && !units.iter().any(|u| u.phones.iter().any(|&(_, s)| s != 0)) {
        for unit in units.iter_mut().rev() {
            if unit.phones.iter().any(|&(p, _)| is_vowel(p)) {
                unit.phones = promote(&unit.phones);
                break;
            }
        }
    }
}

/// Adjust the end of each word to the beginning of the next.
pub fn join_clause(units: &mut [Unit]) {
    for index in 0..units.len().saturating_sub(1) {
        let (head, rest) = units.split_at_mut(index + 1);
        let (unit, following) = (&mut head[index], &rest[0]);
        if unit.phones.is_empty() || following.phones.is_empty() || !is_vowel(following.phones[0].0) {
            continue;
        }
        let n = unit.phones.len();
        let (last, stress) = unit.phones[n - 1];
        if unit.last_word() == "the" && last == "ə" {
            unit.phones[n - 1] = ("ɪ", stress);
        } else if unit.last_word() == "to" && last == "ə" {
            unit.phones[n - 1] = ("ʊ", stress);
        } else if matches!(following.first_word(), "and" | "or" | "if") {
            continue;
        } else if let Some(plain) = split_rhotic(last) {
            unit.phones[n - 1] = (plain, stress);
            unit.phones.push(("ɹ", 0));
        } else if matches!(last, "ɚ" | "ɜː" | "aɪɚ") {
            unit.phones.push(("ɹ", 0));
        } else if last == "t" && matches!(unit.word.as_str(), "it" | "at") && n > 1 && is_vowel(unit.phones[n - 2].0) {
            unit.phones[n - 1] = ("ɾ", 0);
        }
    }
}

/// Units of a clause as spoken words: an attached unit continues the one before.
pub fn groups(units: &[Unit]) -> Vec<Phones> {
    let mut out: Vec<Phones> = Vec::new();
    for unit in units {
        match out.last_mut() {
            Some(last) if unit.attach => last.extend_from_slice(&unit.phones),
            _ => out.push(unit.phones.clone()),
        }
    }
    out.retain(|group| !group.is_empty());
    out
}

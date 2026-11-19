//! Russian G2P: an eSpeak-derived pronunciation lexicon plus letter rules.
//!
//! The Python frontend calls eSpeak NG (GPL, offline). Shipping eSpeak is not
//! allowed, so `scripts/gen/build_g2p_lexicon.py` runs it once over the
//! product vocabulary and writes `word<TAB>phones`. Words outside that lexicon
//! go through the rule fallback below - a small grapheme-to-phoneme model
//! covering palatalisation, vowel reduction and final position.
//!
//! Tokenization mirrors `G2P.tokenize` in `training/voca_ru/text/g2p.py`.

use std::collections::HashMap;
use std::sync::OnceLock;

use fancy_regex::Regex;

use crate::stress::{strip_stress, VOWEL_LETTERS};
use crate::symbols::{is_phone, is_vowel, punctuation, symbol, BOS, EOS, SP, STRESS, UNK};

static LEXICON_TEXT: &[u8] = include_bytes!("data/g2p_lexicon.txt");

type Phone = (&'static str, u8);

fn lexicon() -> &'static HashMap<&'static str, Vec<Phone>> {
    static LEXICON: OnceLock<HashMap<&'static str, Vec<Phone>>> = OnceLock::new();
    LEXICON.get_or_init(|| {
        let text = std::str::from_utf8(LEXICON_TEXT).expect("g2p lexicon is UTF-8");
        let mut map = HashMap::with_capacity(text.lines().count());
        for line in text.lines() {
            let Some((word, phones)) = line.split_once('\t') else {
                continue;
            };
            let parsed = phones
                .split_whitespace()
                .filter_map(|token| {
                    let (name, level) = match token.split_once(':') {
                        Some((name, level)) => (name, level.parse().unwrap_or(0)),
                        None => (token, 0),
                    };
                    symbol(name).map(|s| (s, level))
                })
                .collect();
            map.insert(word, parsed);
        }
        map
    })
}

pub fn lexicon_len() -> usize {
    lexicon().len()
}

/// Vowel index of the combining acute, or `None` when the word is unmarked.
fn target_vowel_index(stressed_word: &str) -> Option<usize> {
    let mut count = 0usize;
    for ch in stressed_word.chars() {
        if ch == STRESS {
            return Some(count.saturating_sub(1));
        }
        if VOWEL_LETTERS.contains(ch.to_lowercase().next().unwrap_or(ch)) {
            count += 1;
        }
    }
    None
}

fn apply_stress(mut phones: Vec<Phone>, target: Option<usize>) -> Vec<Phone> {
    if let Some(target) = target {
        let slots: Vec<usize> = phones
            .iter()
            .enumerate()
            .filter(|(_, (s, _))| is_vowel(s))
            .map(|(i, _)| i)
            .collect();
        if target < slots.len() {
            for phone in phones.iter_mut() {
                phone.1 = 0;
            }
            phones[slots[target]].1 = 1;
        }
    }
    phones
}

const HARD: [(char, &str); 21] = [
    ('б', "b"),
    ('в', "v"),
    ('г', "ɡ"),
    ('д', "d"),
    ('ж', "ʒ"),
    ('з', "z"),
    ('й', "j"),
    ('к', "k"),
    ('л', "ɭ"),
    ('м', "m"),
    ('н', "n"),
    ('п', "p"),
    ('р', "r"),
    ('с', "s"),
    ('т', "t"),
    ('ф', "f"),
    ('х', "x"),
    ('ц', "ts"),
    ('ч', "tʃʲ"),
    ('ш', "ʃ"),
    ('щ', "ɕ"),
];

const SOFT: [(char, &str); 14] = [
    ('б', "bʲ"),
    ('в', "vʲ"),
    ('г', "ɡʲ"),
    ('д', "dʲ"),
    ('з', "ʑ"),
    ('к', "kʲ"),
    ('л', "ɭʲ"),
    ('м', "mʲ"),
    ('н', "nʲ"),
    ('п', "pʲ"),
    ('р', "rʲ"),
    ('с', "sʲ"),
    ('т', "tʲ"),
    ('ф', "fʲ"),
];

fn soft_vowel(c: char) -> bool {
    matches!(c, 'е' | 'ё' | 'и' | 'ю' | 'я')
}

fn consonant(c: char, soft: bool) -> Option<&'static str> {
    if soft {
        if let Some((_, p)) = SOFT.iter().find(|(ch, _)| *ch == c) {
            return Some(p);
        }
    }
    HARD.iter().find(|(ch, _)| *ch == c).map(|(_, p)| *p)
}

/// Rule fallback for words outside the lexicon.
fn rule_word(word: &str) -> Vec<Phone> {
    let plain: Vec<char> = strip_stress(word).chars().collect();
    if plain.is_empty() {
        return Vec::new();
    }
    let vowel_positions: Vec<usize> = plain
        .iter()
        .enumerate()
        .filter(|(_, c)| VOWEL_LETTERS.contains(**c))
        .map(|(i, _)| i)
        .collect();
    let stress_index = target_vowel_index(word)
        .unwrap_or(0)
        .min(vowel_positions.len().saturating_sub(1));
    let mut phones: Vec<Phone> = Vec::new();
    let mut previous_vowel = false;
    let mut previous_soft_sign = false;
    let mut vowel_seen = 0usize;

    for (i, &c) in plain.iter().enumerate() {
        if c == 'ь' || c == 'ъ' {
            previous_soft_sign = c == 'ь';
            previous_vowel = false;
            continue;
        }
        if let Some(phone) = consonant(c, false) {
            let next = plain.get(i + 1).copied().unwrap_or(' ');
            let soft = soft_vowel(next) || next == 'ь';
            phones.push((consonant(c, soft).unwrap_or(phone), 0));
            previous_soft_sign = false;
            previous_vowel = false;
            continue;
        }
        if !VOWEL_LETTERS.contains(c) {
            continue;
        }
        let stressed = vowel_seen == stress_index;
        let position = vowel_positions.iter().position(|p| *p == i).unwrap_or(0);
        let pretonic = stress_index > 0 && position + 1 == stress_index;
        let soft = previous_soft_sign;
        let glide =
            matches!(c, 'е' | 'ё' | 'ю' | 'я') && (i == 0 || previous_vowel || previous_soft_sign);
        if glide {
            phones.push(("j", 0));
        }
        let phone = match c {
            'а' => {
                if stressed {
                    if soft {
                        "a"
                    } else {
                        "ɑ"
                    }
                } else if position + 1 == plain.len() {
                    "a"
                } else if pretonic && !soft {
                    "ʌ"
                } else {
                    "ə"
                }
            }
            'о' => {
                if stressed {
                    "o"
                } else if pretonic {
                    "ʌ"
                } else {
                    "ə"
                }
            }
            'у' => "u",
            'ы' => "y",
            'э' => "ɛ",
            'е' => {
                if stressed {
                    "e"
                } else {
                    "ɪ"
                }
            }
            'и' => {
                if stressed {
                    "i"
                } else {
                    "ɪ"
                }
            }
            'ё' => "ɵ",
            'ю' => "u",
            'я' => {
                if stressed {
                    "a"
                } else {
                    "ɪ"
                }
            }
            _ => continue,
        };
        phones.push((phone, if stressed { 1 } else { 0 }));
        previous_vowel = true;
        previous_soft_sign = false;
        vowel_seen += 1;
    }
    phones
}

fn rules_only() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("VOCA_RU_TEXT_RULES_ONLY").is_some())
}

fn word_phones(stressed_word: &str) -> Vec<Phone> {
    let target = target_vowel_index(stressed_word);
    let plain = strip_stress(stressed_word);
    if !rules_only() {
        if let Some(phones) = lexicon().get(plain.as_str()) {
            return apply_stress(phones.clone(), target);
        }
    }
    if plain.contains('-') {
        let mut out: Vec<Phone> = Vec::new();
        for part in plain.split('-') {
            if out.is_empty() {
                out.extend(word_phones(part));
            } else {
                out.push((SP, 0));
                out.extend(word_phones(part));
            }
        }
        return apply_stress(out, target);
    }
    apply_stress(rule_word(stressed_word), target)
}

enum Piece<'a> {
    Word(&'a str),
    Punct(&'static str),
}

fn is_word_token(token: &str) -> bool {
    token.chars().any(|c| c.is_alphabetic())
        && token
            .chars()
            .all(|c| c.is_alphabetic() || c == '-' || c == STRESS)
}

/// Text -> word / punctuation pieces (mirrors `G2P.pieces`).
fn pieces(text: &str) -> Vec<Piece<'_>> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"[а-яё\u{0301}]+(?:-[а-яё\u{0301}]+)*|[^\sа-яё\u{0301}]").unwrap()
    });
    let mut out = Vec::new();
    for found in re.find_iter(text).flatten() {
        let token = found.as_str();
        if is_word_token(token) {
            out.push(Piece::Word(token));
        } else if let Some(mark) = punctuation(token) {
            out.push(Piece::Punct(mark));
        }
    }
    out
}

/// Stressed text -> (symbols, stress levels).
pub fn phonemize(stressed_text: &str) -> Result<(Vec<&'static str>, Vec<u8>), &'static str> {
    let mut symbols: Vec<&'static str> = vec![BOS];
    let mut stress: Vec<u8> = vec![0];
    let mut has_phone = false;
    for piece in pieces(stressed_text) {
        match piece {
            Piece::Word(token) => {
                if is_phone(symbols.last().copied().unwrap_or(""))
                    || *symbols.last().unwrap() == UNK
                {
                    symbols.push(SP);
                    stress.push(0);
                }
                for (phone, level) in word_phones(token) {
                    if phone == SP {
                        if *symbols.last().unwrap() != SP {
                            symbols.push(SP);
                            stress.push(0);
                        }
                        continue;
                    }
                    if is_phone(phone) || phone == UNK {
                        has_phone = true;
                    }
                    symbols.push(phone);
                    stress.push(level);
                }
            }
            Piece::Punct(mark) => {
                if *symbols.last().unwrap() != mark {
                    symbols.push(mark);
                    stress.push(0);
                }
            }
        }
    }
    symbols.push(EOS);
    stress.push(0);
    if !has_phone {
        return Err("the text has nothing to pronounce");
    }
    Ok((symbols, stress))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexicon_is_embedded() {
        assert!(lexicon_len() > 10_000);
        assert!(lexicon().contains_key("привет"));
    }

    #[test]
    fn privet() {
        let (symbols, stress) = phonemize("приве́т").unwrap();
        assert_eq!(symbols, [BOS, "p", "rʲ", "i", "vʲ", "e", "t", EOS]);
        assert_eq!(stress[symbols.iter().position(|s| *s == "e").unwrap()], 1);
    }

    #[test]
    fn punctuation_and_spacing() {
        let (symbols, _) = phonemize("приве́т, как дела́?").unwrap();
        assert_eq!(
            symbols,
            [
                BOS, "p", "rʲ", "i", "vʲ", "e", "t", ",", "k", "ɑ", "k", SP, "dʲ", "e", "ɭ", "a",
                "?", EOS,
            ]
        );
    }

    #[test]
    fn fallback_rules_are_usable() {
        let phones = rule_word("нейросетевой");
        assert!(phones.iter().any(|(p, _)| *p == "j"));
        assert!(phones.iter().any(|(_, level)| *level == 1));
    }
}

//! The 76-symbol Russian inventory (docs/contracts.md §4).
//!
//! Must stay byte-identical to `training/voca_ru/text/symbols_ru.py`.

pub const SPECIALS: [&str; 5] = ["<pad>", "<bos>", "<eos>", "<sp>", "<unk>"];

pub const PUNCTUATION: [&str; 23] = [
    ",", ".", "?", "!", ";", ":", "\"", "'", "(", ")", "—", "…", "«", "»", "-", "–", "№", "%", "/",
    "[", "]", "{", "}",
];

pub const CONSONANTS: [&str; 36] = [
    "b", "bʲ", "v", "vʲ", "ɡ", "ɡʲ", "d", "dʲ", "ʒ", "z", "ʑ", "j", "k", "kʲ", "ɭ", "ɭʲ", "m",
    "mʲ", "n", "nʲ", "p", "pʲ", "r", "rʲ", "s", "sʲ", "t", "tʲ", "ts", "tʃʲ", "f", "fʲ", "x", "ʃ",
    "ɕ", "dʒ",
];

pub const VOWELS: [&str; 12] = ["ʌ", "a", "ɑ", "ɪ", "i", "ɛ", "e", "ə", "o", "u", "y", "ɵ"];

pub const FIRST_PUNCTUATION: usize = 5;
pub const FIRST_PHONE: usize = 28;
pub const FIRST_VOWEL: usize = 64;

pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";
pub const SP: &str = "<sp>";
pub const UNK: &str = "<unk>";

pub const STRESS: char = '\u{0301}';

/// Index of a symbol in the 76-symbol table.
pub fn symbol_id(symbol: &str) -> Option<usize> {
    let mut idx = 0usize;
    for s in SPECIALS
        .iter()
        .chain(PUNCTUATION.iter())
        .chain(CONSONANTS.iter())
        .chain(VOWELS.iter())
    {
        if *s == symbol {
            return Some(idx);
        }
        idx += 1;
    }
    None
}

/// The inventory's own `'static` copy of any symbol, or `None`.
pub fn symbol(symbol: &str) -> Option<&'static str> {
    SPECIALS
        .iter()
        .chain(PUNCTUATION.iter())
        .chain(CONSONANTS.iter())
        .chain(VOWELS.iter())
        .find(|s| **s == symbol)
        .copied()
}

pub fn is_phone(symbol: &str) -> bool {
    CONSONANTS.iter().chain(VOWELS.iter()).any(|s| *s == symbol)
}

pub fn is_vowel(symbol: &str) -> bool {
    VOWELS.iter().any(|s| *s == symbol)
}

/// The inventory's own copy of a punctuation mark, or `None`.
pub fn punctuation(mark: &str) -> Option<&'static str> {
    PUNCTUATION.iter().find(|s| **s == mark).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_contract() {
        let all: Vec<&str> = SPECIALS
            .iter()
            .chain(PUNCTUATION.iter())
            .chain(CONSONANTS.iter())
            .chain(VOWELS.iter())
            .copied()
            .collect();
        assert_eq!(all.len(), 76);
        assert_eq!(
            all.iter().collect::<std::collections::HashSet<_>>().len(),
            76
        );
        assert_eq!(symbol_id("<bos>"), Some(1));
        assert_eq!(symbol_id("b"), Some(FIRST_PHONE));
        assert_eq!(symbol_id("ʌ"), Some(FIRST_VOWEL));
        assert_eq!(symbol_id("}"), Some(27));
    }
}

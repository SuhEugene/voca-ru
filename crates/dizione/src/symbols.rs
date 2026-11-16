//! The phone inventory and the token sequence built from it.

pub const SYMBOLS: [&str; 76] = [
    "<pad>", "<bos>", "<eos>", "<sp>", "<unk>",
    ",", ".", "?", "!", ";", ":", "\"", "'", "(", ")", "—", "…",
    "p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ",
    "ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
    "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
    "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩",
];
const FIRST_PUNCTUATION: usize = 5;
const FIRST_PHONE: usize = 17;
const FIRST_VOWEL: usize = 43;
pub const BOS: &str = SYMBOLS[1];
pub const EOS: &str = SYMBOLS[2];
pub const SP: &str = SYMBOLS[3];
pub const UNK: &str = SYMBOLS[4];

/// A phone of the inventory with its stress (0 none, 1 primary, 2 secondary).
pub type Ph = (&'static str, u8);
pub type Phones = Vec<Ph>;

pub fn symbol_id(symbol: &str) -> Option<usize> {
    SYMBOLS.iter().position(|s| *s == symbol)
}
/// The inventory's own copy of a phone, `None` for anything else.
pub fn phone(symbol: &str) -> Option<&'static str> {
    SYMBOLS[FIRST_PHONE..].iter().find(|s| **s == symbol).copied()
}
pub fn is_phone(symbol: &str) -> bool {
    phone(symbol).is_some()
}
pub fn is_vowel(symbol: &str) -> bool {
    SYMBOLS[FIRST_VOWEL..].contains(&symbol)
}
/// The inventory's copy of a punctuation mark.
pub fn punctuation(mark: &str) -> Option<&'static str> {
    SYMBOLS[FIRST_PUNCTUATION..FIRST_PHONE].iter().find(|s| **s == mark).copied()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phonemized {
    pub symbols: Vec<&'static str>,
    pub ids: Vec<u32>,
    pub stress: Vec<u8>,
}

impl std::fmt::Display for Phonemized {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, (symbol, stress)) in self.symbols.iter().zip(&self.stress).enumerate() {
            if index > 0 {
                f.write_str(" ")?;
            }
            f.write_str(match stress { 1 => "ˈ", 2 => "ˌ", _ => "" })?;
            f.write_str(symbol)?;
        }
        Ok(())
    }
}

/// One piece of an utterance: a punctuation mark or a run of spoken words.
pub enum Piece {
    Punct(&'static str),
    Words(Vec<Phones>),
}

/// Build the token sequence: words separated by `<sp>`, repeated punctuation collapsed.
pub fn tokenize(pieces: &[Piece]) -> Result<Phonemized, String> {
    let mut symbols: Vec<&'static str> = vec![BOS];
    let mut stress: Vec<u8> = vec![0];
    for piece in pieces {
        match piece {
            Piece::Punct(mark) => {
                if *symbols.last().unwrap() != *mark {
                    symbols.push(mark);
                    stress.push(0);
                }
            }
            Piece::Words(words) => {
                for word in words {
                    let last = *symbols.last().unwrap();
                    if is_phone(last) || last == UNK {
                        symbols.push(SP);
                        stress.push(0);
                    }
                    for &(phone, level) in word {
                        symbols.push(phone);
                        stress.push(level);
                    }
                }
            }
        }
    }
    symbols.push(EOS);
    stress.push(0);
    if !symbols.iter().any(|s| is_phone(s) || *s == UNK) {
        return Err("the text has nothing to pronounce".into());
    }
    let ids = symbols.iter().map(|s| symbol_id(s).unwrap() as u32).collect();
    Ok(Phonemized { symbols, ids, stress })
}

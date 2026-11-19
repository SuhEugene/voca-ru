//! Russian text frontend: normalise -> stress -> phonemes -> token ids.
//!
//! Mirrors `training/voca_ru/text/pipeline.py` without Python, eSpeak or a
//! network at runtime. The 76-symbol inventory and the rules are direct ports;
//! the stress dictionary and the pronunciation lexicon are generated offline
//! (see `scripts/gen/`).

pub mod g2p;
pub mod normalize;
pub mod numbers;
pub mod stress;
pub mod symbols;

use symbols::symbol_id;

#[derive(Debug, Clone)]
pub struct Analysis {
    /// Normalised text without stress marks.
    pub plain_text: String,
    /// Normalised text with one combining acute per stressed word.
    pub stressed_text: String,
    pub phonemes: Vec<&'static str>,
    /// 0 none, 1 primary, 2 secondary, parallel to `phonemes`.
    pub stress: Vec<u8>,
    pub tokens: Vec<usize>,
    pub oov: bool,
}

/// Raw sentence -> every field the reference pipeline produces.
pub fn analyze(text: &str) -> Result<Analysis, &'static str> {
    let plain_text = normalize::normalize(text);
    let (stressed_text, oov) = stress::stressed(&plain_text);
    let (phonemes, stress) = g2p::phonemize(&stressed_text)?;
    let tokens = phonemes
        .iter()
        .map(|symbol| symbol_id(symbol).expect("inventory symbol"))
        .collect();
    Ok(Analysis {
        plain_text,
        stressed_text,
        phonemes,
        stress,
        tokens,
        oov,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end() {
        let result = analyze("Привет, как дела?").unwrap();
        assert_eq!(result.plain_text, "привет, как дела?");
        assert_eq!(result.stressed_text.matches(symbols::STRESS).count(), 3);
        assert_eq!(result.tokens[0], symbol_id("<bos>").unwrap());
        assert_eq!(*result.tokens.last().unwrap(), symbol_id("<eos>").unwrap());
        assert!(result.tokens.iter().all(|id| *id < 76));
    }

    #[test]
    fn digits_never_reach_phonemes() {
        let result = analyze("19 ноября 2026 года, 15:30, 1 500 ₽, 20%.").unwrap();
        assert!(!result.plain_text.contains(|c: char| c.is_ascii_digit()));
        assert!(!result.stressed_text.contains(|c: char| c.is_ascii_digit()));
    }
}

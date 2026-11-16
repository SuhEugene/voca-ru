//! Text to speech: sentence splitting, the dizione phonemizer and the model.

use std::path::Path;
use std::time::{Duration, Instant};

use dizione::{Phonemized, Phonemizer};

use crate::model::{Options, TinyVoca};

pub struct Tts {
    pub model: TinyVoca,
    pub phonemizer: Phonemizer,
}

/// One synthesized utterance (a sentence, or part of a very long one).
pub struct Utterance {
    pub text: String,
    pub symbols: Vec<&'static str>,
    pub samples: usize,
}

pub struct Speech {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    pub utterances: Vec<Utterance>,
    pub phoneme_time: Duration,
    pub model_time: Duration,
}

impl Tts {
    /// `weights` is the safetensors file from `scripts/convert_weights.py`;
    /// `phonemizer_data` the directory with lexicon.tsv, g2p.npz and the rest.
    pub fn load(weights: &Path, phonemizer_data: &Path) -> Result<Self, String> {
        let (model, phonemizer) = rayon::join(|| TinyVoca::load(weights), || Phonemizer::load(phonemizer_data));
        Ok(Self { model: model?, phonemizer: phonemizer? })
    }

    /// Speak text of any length: it is split into sentences, and sentences
    /// too long for one utterance are cut at commas, then between words.
    pub fn speak(&self, text: &str, voice: &str, options: &Options) -> Result<Speech, String> {
        self.speak_parts(sentences(text), voice, options)
    }

    /// Speak the text as a single utterance, as the reference command line
    /// does (an error if it is too long for one).
    pub fn speak_whole(&self, text: &str, voice: &str, options: &Options) -> Result<Speech, String> {
        let started = Instant::now();
        let phones = self.phonemizer.phonemize(text)?;
        if phones.ids.len() > self.model.config.max_tokens {
            return Err("text is too long for one utterance; split it into sentences".into());
        }
        let mut speech = self.empty()?;
        speech.phoneme_time = started.elapsed();
        let voice = self.voice(voice)?;
        let started = Instant::now();
        let result = self.model.synthesize(&phones.ids, &phones.stress, voice, options, None)?;
        speech.model_time = started.elapsed();
        speech.utterances.push(Utterance { text: text.to_string(), symbols: phones.symbols, samples: result.audio.len() });
        speech.audio = result.audio;
        Ok(speech)
    }

    fn voice(&self, voice: &str) -> Result<usize, String> {
        self.model.voice_index(voice).ok_or_else(|| format!("unknown voice {voice}; available: {}", self.model.config.speakers.join(", ")))
    }

    fn empty(&self) -> Result<Speech, String> {
        Ok(Speech { audio: Vec::new(), sample_rate: self.model.config.sample_rate as u32, utterances: Vec::new(),
                    phoneme_time: Duration::ZERO, model_time: Duration::ZERO })
    }

    fn speak_parts(&self, parts: Vec<String>, voice: &str, options: &Options) -> Result<Speech, String> {
        let voice = self.voice(voice)?;
        let mut speech = self.empty()?;
        if parts.is_empty() {
            return Err("the text has nothing to pronounce".into());
        }
        for sentence in parts {
            let started = Instant::now();
            let mut pieces = Vec::new();
            self.fit(&sentence, &mut pieces)?;
            speech.phoneme_time += started.elapsed();
            for (text, phones) in pieces {
                let started = Instant::now();
                let result = self.model.synthesize(&phones.ids, &phones.stress, voice, options, None)?;
                speech.model_time += started.elapsed();
                speech.utterances.push(Utterance { text, symbols: phones.symbols, samples: result.audio.len() });
                speech.audio.extend_from_slice(&result.audio);
            }
        }
        Ok(speech)
    }

    /// Phonemize a sentence, cutting it until every piece fits one utterance.
    fn fit(&self, sentence: &str, out: &mut Vec<(String, Phonemized)>) -> Result<(), String> {
        let phones = self.phonemizer.phonemize(sentence)?;
        if phones.ids.len() <= self.model.config.max_tokens.saturating_sub(12) {
            out.push((sentence.to_string(), phones));
            return Ok(());
        }
        let clauses = split_after(sentence, |c| matches!(c, ',' | ';' | ':' | '—'));
        if clauses.len() > 1 {
            return clauses.iter().try_for_each(|clause| self.fit(clause, out));
        }
        let words: Vec<&str> = sentence.split(' ').collect();
        if words.len() < 2 {
            return Ok(());
        }
        let half = words.len().div_ceil(2);
        self.fit(&words[..half].join(" "), out)?;
        self.fit(&words[half..].join(" "), out)
    }
}

/// Split at spaces that follow a character accepted by `mark`.
fn split_after(text: &str, mark: impl Fn(char) -> bool) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut previous = ' ';
    for c in text.chars() {
        if c == ' ' && mark(previous) {
            parts.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
        previous = c;
    }
    parts.push(current);
    parts.retain(|part| !part.is_empty());
    parts
}

const ABBREVIATIONS: [&str; 16] = ["mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "inc", "ltd", "co", "e.g", "i.e"];
const ABBREVIATIONS_DOTTED: [&str; 2] = ["a.m", "p.m"];

fn is_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// True when the text ends with a title, an initial or a known abbreviation
/// and its full stop, so the stop does not close a sentence.
fn ends_with_abbreviation(text: &str) -> bool {
    let Some(body) = text.strip_suffix('.') else { return false };
    let lower = body.to_ascii_lowercase();
    let boundary = |start: usize| lower[..start].chars().next_back().is_none_or(|c| !is_word(c));
    if ABBREVIATIONS.iter().chain(&ABBREVIATIONS_DOTTED).any(|a| lower.ends_with(a) && boundary(lower.len() - a.len())) {
        return true;
    }
    lower.chars().next_back().is_some_and(|c| c.is_ascii_alphabetic()) && boundary(lower.len() - 1)
}

/// Sentences, without breaking after titles, initials or inside numbers
/// (the rules of the reference web demo).
pub fn sentences(text: &str) -> Vec<String> {
    let text: Vec<char> = text.split_whitespace().collect::<Vec<_>>().join(" ").chars().collect();
    let closes = |c: char| matches!(c, '.' | '!' | '?' | '…');
    let opens_sentence = |c: char| c.is_ascii_uppercase() || c.is_ascii_digit();
    let at = |i: isize| if i >= 0 { text.get(i as usize).copied() } else { None };
    let mut pieces = Vec::new();
    let mut current = String::new();
    for (i, &c) in text.iter().enumerate() {
        let i = i as isize;
        let after_end = at(i - 1).is_some_and(closes) || (at(i - 1).is_some_and(|c| matches!(c, '"' | '\'' | ')' | ']')) && at(i - 2).is_some_and(closes));
        let before_start = at(i + 1).is_some_and(opens_sentence) || (at(i + 1).is_some_and(|c| matches!(c, '"' | '\'' | '(' | '[')) && at(i + 2).is_some_and(opens_sentence));
        if c == ' ' && after_end && before_start {
            pieces.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    pieces.push(current);
    let mut parts = Vec::new();
    let mut current = String::new();
    for piece in pieces {
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(&piece);
        if !ends_with_abbreviation(&current) {
            parts.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts.retain(|part| part.chars().any(char::is_alphanumeric));
    parts
}

#[cfg(test)]
mod tests {
    use super::sentences;

    #[test]
    fn splits_sentences() {
        assert_eq!(sentences("Hello there.  How are you? Fine!"), ["Hello there.", "How are you?", "Fine!"]);
        assert_eq!(sentences("Dr. Smith met Mr. J. Jones at 3.5 p.m. They talked."), ["Dr. Smith met Mr. J. Jones at 3.5 p.m. They talked."]);
        assert_eq!(sentences("He said \"Go.\" Then he left. ... \"Why?\" she asked."), ["He said \"Go.\"", "Then he left. ...", "\"Why?\" she asked."]);
        assert_eq!(sentences("It costs $3.50. 20 people came."), ["It costs $3.50.", "20 people came."]);
        assert!(sentences(" ... ").is_empty());
        assert_eq!(sentences("I said no. Then I left. See e.g. Table 2. OK? (Yes.) [No.] 'Maybe.' Sure… Right."),
                   ["I said no. Then I left.", "See e.g. Table 2.", "OK?", "(Yes.)", "[No.]", "'Maybe.'", "Sure…", "Right."]);
    }
}

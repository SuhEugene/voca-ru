//! Text to speech: Russian sentence splitting, the embedded `voca-ru-text`
//! frontend and the model.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::model::{Options, TinyVoca};

pub struct Tts {
    pub model: TinyVoca,
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

/// Frontend output in the shape the model wants: ids and stress levels.
struct Phones {
    symbols: Vec<&'static str>,
    ids: Vec<u32>,
    stress: Vec<u8>,
}

fn phonemize(text: &str) -> Result<Phones, String> {
    let analysis = voca_ru_text::analyze(text).map_err(str::to_string)?;
    Ok(Phones {
        symbols: analysis.phonemes,
        ids: analysis.tokens.into_iter().map(|id| id as u32).collect(),
        stress: analysis.stress,
    })
}

impl Tts {
    /// `weights` is the safetensors file from `scripts/convert_weights.py`.
    /// The Russian frontend data is embedded in the binary, so
    /// `phonemizer_data` is accepted for compatibility with the installer
    /// layout and ignored.
    pub fn load(weights: &Path, phonemizer_data: &Path) -> Result<Self, String> {
        let _ = phonemizer_data;
        Ok(Self { model: TinyVoca::load(weights)? })
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
        let phones = phonemize(text)?;
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
    fn fit(&self, sentence: &str, out: &mut Vec<(String, Phones)>) -> Result<(), String> {
        let phones = phonemize(sentence)?;
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

// Russian titles, abbreviations and dotted short forms whose full stop does
// not close a sentence.
const ABBREVIATIONS: [&str; 22] = [
    "г", "гг", "др", "им", "ул", "пр", "см", "рис", "стр", "напр", "проф", "акад", "тыс", "млн",
    "млрд", "руб", "коп", "т.е", "т.д", "т.п", "т.к", "т.н",
];

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// True when the text ends with a title, an initial or a known abbreviation
/// and its full stop, so the stop does not close a sentence.
fn ends_with_abbreviation(text: &str) -> bool {
    let Some(body) = text.strip_suffix('.') else { return false };
    let lower = body.to_lowercase();
    let boundary = |start: usize| lower[..start].chars().next_back().is_none_or(|c| !is_word(c));
    if ABBREVIATIONS.iter().any(|a| lower.ends_with(a) && boundary(lower.len() - a.len())) {
        return true;
    }
    lower.chars().next_back().is_some_and(|c| c.is_alphabetic() && boundary(lower.len() - c.len_utf8()))
}

/// Sentences, without breaking after titles, initials or inside numbers.
pub fn sentences(text: &str) -> Vec<String> {
    let text: Vec<char> = text.split_whitespace().collect::<Vec<_>>().join(" ").chars().collect();
    let closes = |c: char| matches!(c, '.' | '!' | '?' | '…');
    let opens_sentence = |c: char| c.is_uppercase() || c.is_numeric();
    let at = |i: isize| if i >= 0 { text.get(i as usize).copied() } else { None };
    let mut pieces = Vec::new();
    let mut current = String::new();
    for (i, &c) in text.iter().enumerate() {
        let i = i as isize;
        let after_end = at(i - 1).is_some_and(closes) || (at(i - 1).is_some_and(|c| matches!(c, '"' | '\'' | ')' | ']' | '»')) && at(i - 2).is_some_and(closes));
        let before_start = at(i + 1).is_some_and(opens_sentence) || (at(i + 1).is_some_and(|c| matches!(c, '"' | '\'' | '(' | '[' | '«')) && at(i + 2).is_some_and(opens_sentence));
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
    use voca_ru_text::symbols::{self, is_phone, is_vowel};

    #[test]
    fn splits_russian_sentences() {
        assert_eq!(sentences("Привет, как дела? Хорошо! Спасибо."), ["Привет, как дела?", "Хорошо!", "Спасибо."]);
        assert_eq!(sentences("Др. Иванов пришёл в 3.5 часа. Потом ушёл."), ["Др. Иванов пришёл в 3.5 часа.", "Потом ушёл."]);
        assert_eq!(sentences("Он сказал «Иди.» Затем ушёл."), ["Он сказал «Иди.»", "Затем ушёл."]);
        assert_eq!(sentences("Это стоит 3,5 рубля. 20 человек пришли."), ["Это стоит 3,5 рубля.", "20 человек пришли."]);
        assert!(sentences(" ... ").is_empty());
    }

    #[test]
    fn frontend_ids_fit_the_model_layout() {
        let analysis = voca_ru_text::analyze("Мой коллега уходит в отпуск на две недели.").unwrap();
        assert_eq!(analysis.tokens.first(), Some(&symbols::symbol_id("<bos>").unwrap()));
        assert_eq!(analysis.tokens.last(), Some(&symbols::symbol_id("<eos>").unwrap()));
        assert!(analysis.tokens.iter().all(|&id| id < symbols::COUNT));
        assert!(analysis.stress.iter().all(|&level| level <= 2));
        for (symbol, &id) in analysis.phonemes.iter().zip(&analysis.tokens) {
            assert_eq!(id, symbols::symbol_id(symbol).unwrap());
            if is_phone(symbol) {
                assert!(id >= symbols::FIRST_PHONE);
            }
            if is_vowel(symbol) {
                assert!(id >= symbols::FIRST_VOWEL);
            }
        }
    }
}

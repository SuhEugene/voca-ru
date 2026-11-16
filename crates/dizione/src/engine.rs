//! The phonemizer: text -> spoken words of phones, clause by clause.
//!
//! Order of authority for one word:
//!   1. pronunciation written inline in the text
//!   2. the built-in exception list (`data/exceptions.txt`); a name of several
//!      words listed there is spoken with the stress written
//!   3. function words and function-word groups (sentence level)
//!   4. the dictionary built from CMUdict
//!   5. regular endings, prefixes and compounds of dictionary words
//!   6. the unknown-word model

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::RwLock;

use crate::align::Aligner;
use crate::convert::{convert, from_text, settle_stress};
use crate::g2p::Model;
use crate::homographs::Homographs;
use crate::morphology;
use crate::normalize::{Normalizer, Token, Word};
use crate::sentence::{self, Unit};
use crate::symbols::{punctuation, Phones, Phonemized, Piece};
use crate::tables::{self, FunctionWord, Phrase, Pronunciation};

/// The dictionary file kept as it is on disk, with an index of its lines.
/// Only the pronunciations that are looked up are ever decoded.
struct Lexicon {
    bytes: Vec<u8>,
    /// (start of the word, end of the word, end of the line), sorted by word.
    lines: Vec<(u32, u32, u32)>,
}

impl Lexicon {
    fn new(bytes: Vec<u8>) -> Self {
        let mut lines: Vec<(u32, u32, u32)> = Vec::with_capacity(bytes.len() / 26);
        let mut sorted = true;
        let mut start = 0;
        for end in memchr::memchr_iter(b'\n', &bytes).chain(std::iter::once(bytes.len())) {
            let mut stop = end;
            if stop > start && bytes[stop - 1] == b'\r' {
                stop -= 1;
            }
            if stop > start {
                if let Some(tab) = memchr::memchr(b'\t', &bytes[start..stop]) {
                    if let Some(&(s, e, _)) = lines.last() {
                        sorted = sorted && bytes[s as usize..e as usize] < bytes[start..start + tab];
                    }
                    lines.push((start as u32, (start + tab) as u32, stop as u32));
                }
            }
            start = end + 1;
            if start > bytes.len() {
                break;
            }
        }
        if !sorted {
            // Stable, so that of two lines for one word the later stays last (and wins, as in a dictionary).
            lines.sort_by(|a, b| bytes[a.0 as usize..a.1 as usize].cmp(&bytes[b.0 as usize..b.1 as usize]));
        }
        Lexicon { bytes, lines }
    }

    fn get(&self, word: &str) -> Option<&str> {
        let bytes = &self.bytes;
        let after = self.lines.partition_point(|&(s, e, _)| &bytes[s as usize..e as usize] <= word.as_bytes());
        if after == 0 {
            return None;
        }
        let (s, e, end) = self.lines[after - 1];
        if &bytes[s as usize..e as usize] == word.as_bytes() { std::str::from_utf8(&bytes[e as usize + 1..end as usize]).ok() } else { None }
    }

    fn contains(&self, word: &str) -> bool {
        let bytes = &self.bytes;
        let after = self.lines.partition_point(|&(s, e, _)| &bytes[s as usize..e as usize] <= word.as_bytes());
        after > 0 && &bytes[self.lines[after - 1].0 as usize..self.lines[after - 1].1 as usize] == word.as_bytes()
    }
}

pub struct Phonemizer {
    lexicon: Lexicon,
    aligner: Aligner,
    model: Option<Model>,
    homographs: Option<Homographs>,
    functions: HashMap<&'static str, FunctionWord>,
    phrases: HashMap<Vec<&'static str>, Phrase>,
    phrase_length: usize,
    exceptions: HashMap<String, Pronunciation>,
    names: HashMap<Vec<String>, Vec<Pronunciation>>,
    name_starts: HashSet<String>,
    name_length: usize,
    letter_names: HashMap<char, Phones>,
    normalizer: Normalizer,
    cache: RwLock<HashMap<String, Phones>>,
}

impl Phonemizer {
    /// Load the built files (`lexicon.tsv`, `align.json`, `g2p.json`,
    /// `g2p.npz`, `homographs.json`) from `data_dir`.
    pub fn load(data_dir: &Path) -> Result<Self, String> {
        let read = |name: &str| std::fs::read_to_string(data_dir.join(name)).map_err(|e| format!("{}: {e}", data_dir.join(name).display()));
        let lexicon = Lexicon::new(std::fs::read(data_dir.join("lexicon.tsv")).map_err(|e| format!("no phonemizer data in {}: {e}", data_dir.display()))?);
        let aligner = Aligner::from_json(&read("align.json")?)?;
        let model = if data_dir.join("g2p.npz").exists() { Some(Model::load(data_dir)?) } else { None };
        let homographs = if data_dir.join("homographs.json").exists() { Some(Homographs::from_json(&read("homographs.json")?)?) } else { None };
        let phrases = tables::phrases()?;
        let names = tables::names(tables::EXCEPTIONS)?;
        let mut letter_names = HashMap::new();
        for (letter, name) in tables::mapping(tables::LETTERS) {
            let mut chars = letter.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                letter_names.insert(c, from_text(name)?);
            }
        }
        Ok(Phonemizer {
            lexicon,
            aligner,
            model,
            homographs,
            functions: tables::function_words()?,
            phrase_length: phrases.keys().map(|k| k.len()).max().unwrap_or(1),
            phrases,
            exceptions: tables::pronunciations(tables::EXCEPTIONS)?,
            name_starts: names.keys().map(|k| k[0].clone()).collect(),
            name_length: names.keys().map(|k| k.len()).max().unwrap_or(1),
            names,
            letter_names,
            normalizer: Normalizer::new(),
            cache: RwLock::new(HashMap::new()),
        })
    }

    // ---- single words -------------------------------------------------

    fn entry(&self, word: &str, entry: &Pronunciation) -> Result<Phones, String> {
        match entry {
            Pronunciation::Phones(phones) => Ok(phones.clone()),
            Pronunciation::Arpabet(symbols) => convert(word, symbols, self.aligner.align(word, symbols).as_deref()),
        }
    }

    /// Pronunciation from the exceptions or the dictionary.
    fn known(&self, word: &str) -> Option<Phones> {
        if let Some(entry) = self.exceptions.get(word) {
            return self.entry(word, entry).ok();
        }
        self.lexicon.get(word).and_then(|text| from_text(text).ok())
    }

    /// Pronunciation of a dictionary word that may be half of a compound: one
    /// that is also listed with a regular ending.
    fn ordinary(&self, word: &str) -> Option<Phones> {
        if word.len() < 3 || !["s", "es", "ed", "ing"].iter().any(|ending| self.lexicon.contains(&format!("{word}{ending}"))) {
            return None;
        }
        self.known(word)
    }

    /// The unknown-word model, then the conversion rules, on one spelling.
    fn guess(&self, word: &str) -> Result<Phones, String> {
        let Some(model) = &self.model else { return Ok(self.spell(word)) };
        let spelling = model.spellable(word);
        let mut symbols = model.predict(&spelling);
        // An r with no vowel beside it is a syllable: write it as one.
        let vowel = |s: &str| s.ends_with(['0', '1', '2']);
        for k in 0..symbols.len() {
            if symbols[k] == "R" && !((k > 0 && vowel(&symbols[k - 1])) || (k + 1 < symbols.len() && vowel(&symbols[k + 1]))) {
                symbols[k] = "ER0".to_string();
            }
        }
        if !symbols.iter().any(|s| vowel(s)) {
            return Ok(self.spell(word));
        }
        let symbols: Vec<&str> = symbols.iter().map(String::as_str).collect();
        // Every word spoken alone has a main stress, even when the model marked none.
        Ok(sentence::promote(&convert(&spelling, &symbols, self.aligner.align(&spelling, &symbols).as_deref())?))
    }

    /// Letter names, joined: the last letter carries the main stress.
    fn spell(&self, letters: &str) -> Phones {
        let lower = letters.to_lowercase();
        let possessive = lower.ends_with("'s");
        let body = if possessive { &lower[..lower.len() - 2] } else { &lower[..] };
        let names: Vec<&Phones> = body.chars().filter_map(|c| self.letter_names.get(&c)).collect();
        let mut out = Vec::new();
        for (index, name) in names.iter().enumerate() {
            out.extend(name.iter().map(|&(phone, stress)| (phone, if stress != 0 && index < names.len() - 1 { 2 } else { stress })));
        }
        if possessive && !out.is_empty() { morphology::plural(&out) } else { out }
    }

    /// True for a spelling the tables or the dictionary know.
    pub fn is_word(&self, word: &str) -> bool {
        self.lexicon.contains(word) || self.functions.contains_key(word) || self.exceptions.contains_key(word)
    }

    /// Phones for one spelling in lower case.
    fn word(&self, word: &str) -> Result<Phones, String> {
        if let Some(found) = self.cache.read().unwrap().get(word) {
            return Ok(found.clone());
        }
        let mut phones = self.known(word);
        if phones.is_none() {
            phones = morphology::derive(word, &|w| self.known(w), &|w| self.ordinary(w), 0).map(|p| settle_stress(&p));
        }
        if phones.is_none() && word.ends_with("'s") && word.len() > 3 {
            let stem = self.guess(&word[..word.len() - 2])?;
            if stem.is_empty() {
                return Err(format!("nothing to pronounce in {word:?}"));
            }
            phones = Some(morphology::plural(&stem));
        }
        let phones = match phones {
            Some(phones) => phones,
            None => self.guess(word)?,
        };
        self.cache.write().unwrap().insert(word.to_string(), phones.clone());
        Ok(phones)
    }

    // ---- sentences ----------------------------------------------------

    /// Units of one clause. `written` is the whole text as tokens and
    /// `places` where each word sits in it, for the homograph rules.
    fn units(&self, words: &[&Word], places: &[usize], written: &[&str]) -> Result<Vec<Unit>, String> {
        let mut units = Vec::with_capacity(words.len());
        let mut index = 0;
        let unit = |word: &str, phones: Phones, kind: &'static str, last: Option<Phones>, attach: bool| Unit { word: word.to_string(), phones, kind, last, attach };
        while index < words.len() {
            let token = words[index];
            if let Some(phones) = &token.phones {
                units.push(unit(&token.text, phones.clone(), "p", None, token.attach));
            } else if token.spell {
                units.push(unit(&token.text, self.spell(&token.text), "p", None, token.attach));
            } else {
                let size = if self.name_starts.contains(&token.text) { self.name(words, index) } else { 0 };
                if size > 0 {
                    let key: Vec<String> = words[index..index + size].iter().map(|w| w.text.clone()).collect();
                    for (position, (text, entry)) in key.iter().zip(&self.names[&key]).enumerate() {
                        units.push(unit(text, self.entry(text, entry)?, "p", None, token.attach && position == 0));
                    }
                    index += size;
                    continue;
                }
                let size = self.phrase(words, index);
                if size > 0 {
                    let key: Vec<&str> = words[index..index + size].iter().map(|w| w.text.as_str()).collect();
                    let (kind, forms, last) = &self.phrases[key.as_slice()];
                    if forms.len() == 1 {
                        units.push(unit(&key.join(" "), forms[0].clone(), kind, last.clone(), token.attach));
                    } else {
                        for (position, (text, form)) in key.iter().zip(forms).enumerate() {
                            units.push(unit(text, form.clone(), kind, None, token.attach && position == 0));
                        }
                    }
                    index += size;
                    continue;
                }
                let text = token.text.as_str();
                match &self.homographs {
                    Some(homographs) if !written.is_empty() && homographs.covers(text) && !self.exceptions.contains_key(text) => {
                        let (phones, ending) = homographs.pronounce(text, written, places[index]);
                        let phones = from_text(phones)?;
                        let phones = if ending {
                            if phones.is_empty() {
                                return Err(format!("nothing to pronounce in {text:?}"));
                            }
                            morphology::plural(&phones)
                        } else {
                            phones
                        };
                        units.push(unit(text, phones, "c", None, token.attach));
                    }
                    _ => {
                        if let Some((kind, form, last)) = self.functions.get(text) {
                            units.push(unit(text, form.clone(), kind, last.clone(), token.attach));
                        } else {
                            units.push(unit(text, self.word(text)?, "c", None, token.attach));
                        }
                    }
                }
            }
            index += 1;
        }
        Ok(units)
    }

    fn name(&self, words: &[&Word], index: usize) -> usize {
        let longest = self.name_length.min(words.len() - index);
        for size in (2..=longest).rev() {
            let span = &words[index..index + size];
            let key: Vec<String> = span.iter().map(|w| w.text.clone()).collect();
            if self.names.contains_key(&key) && span.iter().all(|w| w.phones.is_none() && !w.spell) && !span[1..].iter().any(|w| w.attach) {
                return size;
            }
        }
        0
    }

    fn phrase(&self, words: &[&Word], index: usize) -> usize {
        let longest = self.phrase_length.min(words.len() - index);
        for size in (2..=longest).rev() {
            let span = &words[index..index + size];
            if span[1..].iter().all(|w| w.phones.is_none() && !w.spell && !w.attach) && span[0].phones.is_none() && !span[0].spell {
                let key: Vec<&str> = span.iter().map(|w| w.text.as_str()).collect();
                if self.phrases.contains_key(key.as_slice()) {
                    return size;
                }
            }
        }
        0
    }

    /// Punctuation marks and runs of spoken words, in order.
    pub fn pieces(&self, text: &str) -> Result<Vec<Piece>, String> {
        let stream = self.normalizer.tokens(text, &|word| self.is_word(word))?;
        // The text as the homograph rules see it; a word given only as phones stands as "x".
        let written: Vec<&str> = stream
            .iter()
            .map(|token| match token {
                Token::Punct(mark) => *mark,
                Token::Word(word) => if !word.raw.is_empty() { word.raw.as_str() } else if !word.text.is_empty() { word.text.as_str() } else { "x" },
            })
            .collect();
        let mut out = Vec::new();
        let mut words: Vec<&Word> = Vec::new();
        let mut places: Vec<usize> = Vec::new();
        let flush = |words: &mut Vec<&Word>, places: &mut Vec<usize>, out: &mut Vec<Piece>| -> Result<(), String> {
            if !words.is_empty() {
                let mut units = self.units(words, places, &written)?;
                sentence::stress_clause(&mut units);
                sentence::join_clause(&mut units);
                out.push(Piece::Words(sentence::groups(&units)));
                words.clear();
                places.clear();
            }
            Ok(())
        };
        for (place, token) in stream.iter().enumerate() {
            match token {
                Token::Punct(mark) => {
                    flush(&mut words, &mut places, &mut out)?;
                    out.push(Piece::Punct(punctuation(mark).ok_or("unknown punctuation mark")?));
                }
                Token::Word(word) => {
                    words.push(word);
                    places.push(place);
                }
            }
        }
        flush(&mut words, &mut places, &mut out)?;
        Ok(out)
    }

    /// Text -> tokens: symbols, ids and stress, as `dizione.phonemize(text)` gives them.
    pub fn phonemize(&self, text: &str) -> Result<Phonemized, String> {
        if crate::pyunicode::strip(text).is_empty() {
            return Err("text must not be empty".into());
        }
        crate::symbols::tokenize(&self.pieces(text)?)
    }
}

//! One word: CMUdict pronunciation -> phones of the inventory (the citation
//! form). Flaps, glottal stops, r-coloured vowels, syllabic l and the reduced
//! vowels are added by rule, many of them from how the vowel is spelled.

use crate::align::base;
use crate::symbols::{is_vowel, phone as inventory, Ph, Phones};

const ARPA_VOWELS: [&str; 15] = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"];
const CONSONANTS: [(&str, &str); 24] = [
    ("B", "b"), ("CH", "tʃ"), ("D", "d"), ("DH", "ð"), ("F", "f"), ("G", "ɡ"), ("HH", "h"), ("JH", "dʒ"), ("K", "k"),
    ("L", "l"), ("M", "m"), ("N", "n"), ("NG", "ŋ"), ("P", "p"), ("R", "ɹ"), ("S", "s"), ("SH", "ʃ"), ("T", "t"),
    ("TH", "θ"), ("V", "v"), ("W", "w"), ("Y", "j"), ("Z", "z"), ("ZH", "ʒ"),
];
const FULL_VOWELS: [(&str, &str); 15] = [
    ("AA", "ɑː"), ("AE", "æ"), ("AH", "ʌ"), ("AO", "ɔː"), ("AW", "aʊ"), ("AY", "aɪ"), ("EH", "ɛ"), ("ER", "ɜː"),
    ("EY", "eɪ"), ("IH", "ɪ"), ("IY", "iː"), ("OW", "oʊ"), ("OY", "ɔɪ"), ("UH", "ʊ"), ("UW", "uː"),
];
const SIBILANTS: [&str; 6] = ["S", "Z", "SH", "ZH", "CH", "JH"];
const FORCE_SPELLINGS: [&str; 10] = ["forc", "ford", "sword", "worn", "torn", "sworn", "borne", "shorn", "pork", "divorc"];
const PREFIXES: [&str; 8] = ["dis", "mis", "un", "in", "re", "non", "over", "pre"];
const WEAK_PREFIXES: [&str; 4] = ["be", "de", "re", "pre"];
const IRE_ENDINGS: [&str; 8] = ["ing", "ings", "ingly", "able", "y", "er", "ers", "est"];

fn arpa_vowel(name: &str) -> bool {
    ARPA_VOWELS.contains(&name)
}
fn consonant(name: &str) -> Option<&'static str> {
    CONSONANTS.iter().find(|(k, _)| *k == name).map(|(_, v)| *v)
}
fn full_vowel(name: &str) -> &'static str {
    FULL_VOWELS.iter().find(|(k, _)| *k == name).map(|(_, v)| *v).unwrap()
}
fn vowel_letter(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}
fn ph(symbol: &str) -> &'static str {
    inventory(symbol).unwrap()
}

/// True for a non-empty list of CMUdict symbols.
pub fn is_arpabet<S: AsRef<str>>(symbols: &[S]) -> bool {
    !symbols.is_empty()
        && symbols.iter().all(|s| {
            let s = s.as_ref();
            consonant(s).is_some()
                || (s.len() >= 2 && s.is_char_boundary(s.len() - 1) && matches!(&s[s.len() - 1..], "0" | "1" | "2") && arpa_vowel(&s[..s.len() - 1]))
        })
}

/// `"h ə l ˈoʊ"` -> phones; an error for a symbol outside the inventory.
pub fn from_text(text: &str) -> Result<Phones, String> {
    let mut phones = Vec::new();
    for unit in crate::pyunicode::split(text) {
        let (stress, name) = if let Some(rest) = unit.strip_prefix('ˈ') {
            (1, rest)
        } else if let Some(rest) = unit.strip_prefix('ˌ') {
            (2, rest)
        } else {
            (0, unit)
        };
        let phone = inventory(name).ok_or_else(|| format!("{name:?} is not a phone of the inventory"))?;
        if stress != 0 && !is_vowel(phone) {
            return Err(format!("stress mark on the consonant {phone:?}"));
        }
        phones.push((phone, stress));
    }
    Ok(phones)
}

/// The aligned pronunciation of one word and the questions the rules ask of it.
struct Word<'a> {
    word: &'a str,
    chars: Vec<char>,
    bases: Vec<&'a str>,
    stress: Vec<Option<u8>>,
    aligned: bool,
    runs: Vec<String>,
    starts: Vec<usize>,
    count: usize,
}

impl<'a> Word<'a> {
    fn new(word: &'a str, arpabet: &[&'a str], runs: Option<&[String]>) -> Result<Self, String> {
        let bases: Vec<&str> = arpabet.iter().map(|s| base(s)).collect();
        let stress = arpabet.iter().map(|s| s.chars().next_back().and_then(|c| c.to_digit(10)).filter(|d| *d <= 2).map(|d| d as u8)).collect();
        for (symbol, name) in arpabet.iter().zip(&bases) {
            if consonant(name).is_none() && !arpa_vowel(name) {
                return Err(format!("{symbol:?} is not an ARPABET symbol"));
            }
        }
        let aligned = runs.is_some();
        let runs: Vec<String> = match runs {
            Some(runs) => runs.to_vec(),
            None => vec![String::new(); arpabet.len()],
        };
        let mut starts = Vec::with_capacity(runs.len());
        let mut position = 0;
        for run in &runs {
            starts.push(position);
            position += run.len();
        }
        let count = bases.len();
        Ok(Word { word, chars: word.chars().collect(), bases, stress, aligned, runs, starts, count })
    }
    fn base(&self, i: isize) -> Option<&str> {
        if i >= 0 && (i as usize) < self.count { Some(self.bases[i as usize]) } else { None }
    }
    fn is(&self, i: isize, name: &str) -> bool {
        self.base(i) == Some(name)
    }
    fn among(&self, i: isize, names: &[&str]) -> bool {
        self.base(i).is_some_and(|b| names.contains(&b))
    }
    fn vowel(&self, i: isize) -> bool {
        self.base(i).is_some_and(arpa_vowel)
    }
    fn unstressed(&self, i: isize) -> bool {
        self.vowel(i) && self.stress[i as usize].unwrap_or(0) == 0
    }
    fn letters(&self, i: usize) -> String {
        self.runs[i].chars().filter(|c| vowel_letter(*c)).collect()
    }
    /// Spelling that precedes the letters of segment `i`.
    fn before(&self, i: usize) -> &str {
        if self.aligned { &self.word[..self.starts[i]] } else { "" }
    }
    /// Spelling that follows the letters of segment `i`.
    fn after(&self, i: usize) -> &str {
        if self.aligned { &self.word[self.starts[i] + self.runs[i].len()..] } else { "" }
    }
    /// True when exactly `symbols` follow segment `i` to the end of the word.
    fn ends_with(&self, i: usize, symbols: &[&str]) -> bool {
        self.bases[(i + 1).min(self.count)..] == *symbols
    }
    fn first_vowel(&self, i: usize) -> bool {
        !self.bases[..i].iter().any(|b| arpa_vowel(b))
    }
    fn next_vowel_stress(&self, i: usize) -> u8 {
        for j in i + 1..self.count {
            if arpa_vowel(self.bases[j]) {
                return self.stress[j].unwrap_or(0);
            }
        }
        0
    }
    /// Segment `i` is the "e" of an unstressed be-, de-, re- or pre-.
    fn weak_prefix(&self, i: usize) -> bool {
        self.aligned
            && self.first_vowel(i)
            && self.letters(i) == "e"
            && WEAK_PREFIXES.iter().any(|p| self.word.starts_with(p))
            && matches!(self.before(i), "b" | "d" | "r" | "pr")
    }
    fn char_at(&self, i: usize) -> Option<char> {
        self.chars.get(i).copied()
    }
}

/// AO + R: `"oː"` for the FORCE set (more, four, port), `"ɔː"` for NORTH (for, short).
fn north_or_force(w: &Word, i: usize, before_vowel: bool) -> &'static str {
    if !w.aligned {
        return "ɔː";
    }
    let spelled = w.letters(i);
    if matches!(spelled.as_str(), "oa" | "oo" | "ou") {
        return "oː";
    }
    if spelled != "o" {
        return "ɔː";
    }
    let run = &w.runs[i];
    let cut = run.find('o').map_or(0, |p| p + 1);
    let tail = format!("{}{}", &run[cut..], w.after(i));
    let rest: String = tail.chars().skip(1).collect();
    if !before_vowel {
        if rest.starts_with('e') {
            return "oː";
        }
        if w.before(i).ends_with('p') && rest.starts_with('t') {
            return "oː";
        }
        return if FORCE_SPELLINGS.iter().any(|s| w.word.contains(s)) { "oː" } else { "ɔː" };
    }
    if rest.starts_with('r') {
        return "ɔː";
    }
    if matches!(rest.as_str(), "y" | "ies" | "ied" | "a" | "as" | "ah" | "er" | "ers")
        || ["ia", "io", "iu", "ing", "age"].iter().any(|p| rest.starts_with(p))
    {
        return "oː";
    }
    "ɔː"
}

/// A reduced vowel written "i".
fn weak_i(w: &Word, i: usize) -> &'static str {
    let s = i as isize;
    if w.is(s + 1, "F") && w.is(s + 2, "AH") && w.is(s + 3, "L") {
        return "i";
    }
    if (w.is(s + 1, "T") && w.is(s + 2, "IY")) || (w.is(s + 1, "B") && w.among(s + 2, &["AH", "L"])) || (w.is(s + 1, "F") && w.is(s + 2, "AY")) {
        return "ᵻ";
    }
    if w.is(s + 1, "T") && w.ends_with(i, &["T"]) && w.among(s - 1, &["R", "ER"]) && w.word.ends_with("ite") {
        return "ᵻ";
    }
    "ɪ"
}

/// A reduced vowel written "e" outside a weak prefix.
fn weak_e(w: &Word, i: usize) -> &'static str {
    let s = i as isize;
    if w.is(s + 1, "T") && (w.ends_with(i, &["T"]) || w.ends_with(i, &["T", "S"])) {
        return if w.is(s - 1, "L") { "ə" } else if w.is(s - 1, "R") { "ᵻ" } else { "ɪ" };
    }
    if w.is(s + 1, "S") && (w.ends_with(i, &["S", "T"]) || w.ends_with(i, &["S", "T", "L", "IY"])) {
        return "ɪ";
    }
    if w.is(s + 1, "JH") {
        return "ɪ";
    }
    if w.is(s + 1, "K") && w.is(s + 2, "T") {
        return "ɛ";
    }
    if w.is(s + 1, "D") && w.ends_with(i, &["D"]) {
        return "ɪ";
    }
    "ə"
}

/// The vowel of "-ed" after t/d and of "-es" after a sibilant.
fn inflection(w: &Word, i: usize) -> bool {
    if i + 1 >= w.count || (w.aligned && !w.runs[i].contains('e') && !w.runs[i + 1].contains('e')) {
        return false;
    }
    let s = i as isize;
    if w.is(s + 1, "D") && w.among(s - 1, &["T", "D"]) && (w.ends_with(i, &["D"]) || w.ends_with(i, &["D", "L", "IY"])) {
        return true;
    }
    w.is(s + 1, "Z") && w.among(s - 1, &SIBILANTS) && w.ends_with(i, &["Z"])
}

/// An unstressed first vowel right before the stressed syllable.
fn first_syllable(w: &Word, i: usize) -> Option<&'static str> {
    if !w.aligned || !w.first_vowel(i) || w.next_vowel_stress(i) == 0 {
        return None;
    }
    let word = w.word;
    if i == 0 && word.starts_with('e') {
        if word.starts_with("ex") {
            return Some("ɛ");
        }
        if let (Some(c1), Some(c2)) = (w.char_at(1), w.char_at(2)) {
            if matches!(c1, 'n' | 'm' | 's') && !vowel_letter(c2) && c2 != c1 {
                return Some("ɛ");
            }
        }
        return Some("ɪ");
    }
    if w.weak_prefix(i) {
        if word.starts_with("pre") || (word.starts_with("be") && matches!(w.char_at(2), Some('c') | Some('g'))) {
            return Some("ɪ");
        }
        return Some("ᵻ");
    }
    if w.letters(i) == "i" && word.starts_with("di") && w.chars.len() > 3 && !vowel_letter(w.chars[2]) && w.chars[2] != 's' && vowel_letter(w.chars[3]) {
        return Some("ᵻ");
    }
    None
}

/// Quality of an unstressed AH or IH.
fn reduced(w: &Word, i: usize) -> &'static str {
    let name = w.bases[i];
    let spelled = w.letters(i);
    let spelled = spelled.as_str();
    let s = i as isize;
    if name == "IH" && w.is(s + 1, "NG") {
        return "ɪ";
    }
    if inflection(w, i) {
        return "ᵻ";
    }
    if name == "AH" {
        if w.is(s - 1, "Y") && (w.runs[i - 1].contains('u') || w.runs[i].contains('u')) {
            return "ʊ";
        }
        if i == 0 && w.word.starts_with('a') {
            return "ɐ";
        }
        if i == 0 && w.is(s + 1, "N") && w.word.starts_with("un") {
            return "ʌ";
        }
        if spelled == "a" && PREFIXES.contains(&w.before(i)) && w.runs[i].starts_with('a') {
            return "ɐ";
        }
    }
    if let Some(prefix) = first_syllable(w, i) {
        return prefix;
    }
    if spelled == "e" && w.among(s + 1, &["K", "G"]) && w.runs[i + 1].starts_with('x') && w.next_vowel_stress(i) != 0 {
        return "ɛ";
    }
    if matches!(spelled, "i" | "ai" | "ui" | "y") {
        return weak_i(w, i);
    }
    if spelled == "e" {
        return weak_e(w, i);
    }
    if spelled == "a" {
        if w.is(s + 1, "JH") || (w.is(s + 1, "N") && w.is(s + 2, "JH")) {
            return "ɪ";
        }
        if w.is(s + 1, "S") && w.ends_with(i, &["S"]) && w.word.ends_with("ace") {
            return "ɪ";
        }
        if name == "IH" && !(w.is(s + 1, "T") && w.ends_with(i, &["T"])) {
            return "ɪ";
        }
        return "ə";
    }
    if name == "AH" && w.is(s + 1, "M") && w.ends_with(i, &["M"]) && w.word.ends_with("ome") {
        return "ʌ";
    }
    if name == "IH" && !matches!(spelled, "o" | "u" | "ou") { "ɪ" } else { "ə" }
}

fn unstressed_iy(w: &Word, i: usize) -> Ph {
    let s = i as isize;
    if w.base(s + 1).is_none() || w.ends_with(i, &["Z"]) || w.ends_with(i, &["D"]) {
        return ("i", 0);
    }
    if w.vowel(s + 1) {
        return (if w.is(s + 1, "IH") && w.is(s + 2, "NG") { "i" } else { "ɪ" }, 0);
    }
    if w.weak_prefix(i) {
        return if w.next_vowel_stress(i) == 1 { ("ᵻ", 0) } else { ("iː", 2) };
    }
    ("ɪ", 0)
}

/// Citation form of `word`. `arpabet` is a CMUdict pronunciation and `runs`
/// the letters aligned to each symbol, or `None`.
pub fn convert(word: &str, arpabet: &[&str], runs: Option<&[String]>) -> Result<Phones, String> {
    let w = Word::new(word, arpabet, runs)?;
    let mut out: Phones = Vec::with_capacity(w.count + 2);
    let mut i = 0usize;
    while i < w.count {
        let name = w.bases[i];
        let stress = w.stress[i].unwrap_or(0);
        let s = i as isize;
        let nxt = w.base(s + 1);
        if !arpa_vowel(name) {
            if name == "Y"
                && nxt == Some("AH")
                && w.unstressed(s + 1)
                && w.among(s - 1, &["N", "L"])
                && !w.is(s + 2, "L")
                && (!w.aligned || w.runs[i].contains('i') || w.runs[i + 1].contains('i'))
            {
                out.push(("iə", 0));
                i += 2;
                continue;
            }
            out.push((consonant(name).unwrap(), 0));
            i += 1;
            continue;
        }
        let before_vowel = w.vowel(s + 2);
        let n_r = nxt == Some("R");
        if name == "ER" {
            out.push((if stress != 0 { "ɜː" } else { "ɚ" }, stress));
            if n_r {
                i += 1;
            }
            if w.vowel(i as isize + 1) {
                out.push(("ɹ", 0));
            }
        } else if n_r && name == "AW" {
            out.push(("aʊ", stress));
            out.push(("ɚ", 0));
            if before_vowel {
                out.push(("ɹ", 0));
            }
            i += 1;
        } else if n_r && matches!(name, "AA" | "AO" | "IH" | "IY" | "EH" | "UH" | "UW" | "AY") {
            let (quality, single): (&'static str, &'static str) = match name {
                "AA" => ("ɑː", "ɑːɹ"),
                "AO" => {
                    let q = north_or_force(&w, i, before_vowel);
                    (q, if q == "oː" { "oːɹ" } else { "ɔːɹ" })
                }
                "IH" | "IY" => (if stress != 0 && matches!(w.letters(i).as_str(), "e" | "eo") { "iə" } else { "ɪ" }, "ɪɹ"),
                "EH" => {
                    let joined = format!("{}{}", w.runs[i + 1], w.after(i + 1));
                    let doubled = joined.starts_with("rr") || (w.runs[i].ends_with('r') && w.runs[i + 1].starts_with('r'));
                    (if w.letters(i) == "a" && doubled { "æ" } else { "ɛ" }, "ɛɹ")
                }
                "UH" | "UW" => ("ʊɹ", "ʊɹ"),
                _ => (if !w.aligned || IRE_ENDINGS.contains(&w.after(i + 1)) { "aɪɚ" } else { "aɪ" }, "aɪɚ"),
            };
            if before_vowel {
                out.push((quality, stress));
                out.push(("ɹ", 0));
            } else {
                out.push((single, stress));
            }
            i += 1;
        } else if name == "AY" && nxt == Some("ER") && w.unstressed(s + 1) && w.runs[i + 1] == "er" && w.runs[i] != "i" {
            out.push(("aɪ", stress));
        } else if name == "AY" && nxt == Some("ER") && w.unstressed(s + 1) {
            out.push(("aɪɚ", stress));
            i += 1;
            if w.vowel(i as isize + 1) {
                out.push(("ɹ", 0));
            }
        } else if name == "AY" && nxt == Some("AH") && w.unstressed(s + 1) {
            out.push(("aɪə", stress));
            i += 1;
        } else if name == "IY" && nxt == Some("AH") && w.unstressed(s + 1) && !w.weak_prefix(i) {
            if stress != 1 && (w.among(s + 2, &["L", "D", "T"]) || (w.is(s + 2, "N") && w.is(s + 3, "S"))) {
                out.push(("ɪ", 0));
            } else {
                out.push(("iə", stress));
                i += 1;
            }
        } else if stress == 0
            && name == "AH"
            && nxt == Some("N")
            && w.is(s - 1, "T")
            && !w.is(s + 2, "T")
            && out.len() >= 2
            && out[out.len() - 1].0 == "t"
            && is_vowel(out[out.len() - 2].0)
            && out[out.len() - 2].1 != 0
        {
            let last = out.len() - 1;
            out[last].0 = "ʔ";
            out.push(("n̩", 0));
            i += 1;
        } else if stress == 0
            && name == "AH"
            && nxt == Some("W")
            && w.is(s + 2, "AH")
            && w.unstressed(s + 2)
            && (w.runs[i].contains('u') || w.runs[i + 1].contains('u'))
        {
            out.push(("uː", 0));
            i += 1;
        } else if stress == 0 && name == "AH" && nxt == Some("L") && i > 0 && !(w.vowel(s + 2) && w.stress[i + 2].unwrap_or(0) != 0) {
            if w.letters(i) == "i" && w.vowel(s + 2) {
                out.push(("ɪ", 0));
            } else if w.is(s - 1, "Y") && (w.runs[i - 1].contains('u') || w.runs[i].contains('u')) {
                out.push(("ʊ", 0));
            } else {
                out.push(("əl", 0));
                i += 1;
            }
        } else if stress == 0 && matches!(name, "AH" | "IH") {
            out.push((reduced(&w, i), 0));
        } else if name == "IY" && (stress == 0 || (stress == 2 && nxt.is_none())) {
            out.push(unstressed_iy(&w, i));
        } else if stress == 0 && name == "AE" && i == 0 && (w.word.starts_with("ad") || w.word.starts_with("acc")) && matches!(nxt, Some("D") | Some("K")) {
            out.push(("ɐ", 0));
        } else if stress == 0 && name == "ER" {
            out.push(("ɚ", 0));
        } else if matches!(name, "AA" | "AO") && w.letters(i) == "a" && (w.before(i).ends_with('w') || w.before(i).ends_with("qu")) && nxt == Some("N") {
            out.push(("ɔ", stress));
        } else if name == "AO" && w.letters(i) == "o" {
            out.push(("ɔ", stress));
        } else {
            out.push((full_vowel(name), stress));
        }
        i += 1;
    }
    flap(&mut out);
    for k in 0..out.len().saturating_sub(1) {
        if out[k].0 == "n" && matches!(out[k + 1].0, "k" | "ɡ") {
            out[k].0 = "ŋ";
        }
    }
    settle(&mut out);
    Ok(out.into_iter().map(|(p, s)| (ph(p), s)).collect())
}

/// t between a vowel (or r) and an unstressed vowel: water, city, party.
fn flap(out: &mut Phones) {
    let n = out.len();
    for k in 1..n.saturating_sub(1) {
        if out[k].0 != "t" {
            continue;
        }
        let before = out[k - 1].0;
        let (after, stress) = out[k + 1];
        if (!is_vowel(before) && before != "ɹ") || matches!(before, "n̩" | "əl") || !is_vowel(after) || after == "n̩" {
            continue;
        }
        let final_o = after == "oʊ" && (k + 2 == n || (k + 3 == n && out[k + 2].0 == "z"));
        if stress != 0 && !final_o {
            continue;
        }
        if matches!(after, "ə" | "ɚ") && k + 2 < n && out[k + 2].0 == "n" {
            continue;
        }
        out[k].0 = "ɾ";
    }
}

/// One primary stress per word, and no secondary stress next to the primary one.
pub fn settle(out: &mut Phones) {
    let vowels: Vec<usize> = (0..out.len()).filter(|&k| is_vowel(out[k].0)).collect();
    let primaries: Vec<usize> = vowels.iter().copied().filter(|&k| out[k].1 == 1).collect();
    if let Some((_, rest)) = primaries.split_last() {
        for &k in rest {
            out[k].1 = 2;
        }
    }
    for (position, &k) in vowels.iter().enumerate() {
        if out[k].1 != 2 {
            continue;
        }
        let near_primary = (position > 0 && out[vowels[position - 1]].1 == 1) || (position + 1 < vowels.len() && out[vowels[position + 1]].1 == 1);
        if near_primary {
            out[k].1 = 0;
        }
    }
}

pub fn settle_stress(phones: &[Ph]) -> Phones {
    let mut out = phones.to_vec();
    settle(&mut out);
    out
}

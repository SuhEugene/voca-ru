//! Words the dictionary lacks but can be built from words it has: regular
//! endings, a few prefixes, compounds of two dictionary words, British spellings.

use crate::symbols::{is_vowel, Ph, Phones};

const SIBILANT: [&str; 6] = ["s", "z", "ʃ", "ʒ", "tʃ", "dʒ"];
const VOICELESS: [&str; 6] = ["p", "t", "k", "f", "θ", "ʔ"];
const RHOTIC: [&str; 4] = ["ɚ", "ɜː", "aɪɚ", "ʊɹ"];

fn split_rhotic(phone: &str) -> Option<&'static str> {
    Some(match phone {
        "ɑːɹ" => "ɑː",
        "ɔːɹ" => "ɔː",
        "oːɹ" => "oː",
        "ɪɹ" => "ɪ",
        "ɛɹ" => "ɛ",
        _ => return None,
    })
}
fn consonant_letter(c: char) -> bool {
    "bcdfghjklmnpqrstvwxz".contains(c)
}

// Endings that leave the stem's stress alone: spelling, phones.
const ENDINGS: [(&str, &[Ph]); 22] = [
    ("fully", &[("f", 0), ("əl", 0), ("i", 0)]),
    ("lessly", &[("l", 0), ("ə", 0), ("s", 0), ("l", 0), ("i", 0)]),
    ("ness", &[("n", 0), ("ə", 0), ("s", 0)]),
    ("less", &[("l", 0), ("ə", 0), ("s", 0)]),
    ("ments", &[("m", 0), ("ə", 0), ("n", 0), ("t", 0), ("s", 0)]),
    ("ment", &[("m", 0), ("ə", 0), ("n", 0), ("t", 0)]),
    ("ful", &[("f", 0), ("əl", 0)]),
    ("ship", &[("ʃ", 0), ("ɪ", 2), ("p", 0)]),
    ("hood", &[("h", 0), ("ʊ", 2), ("d", 0)]),
    ("like", &[("l", 0), ("aɪ", 2), ("k", 0)]),
    ("dom", &[("d", 0), ("ə", 0), ("m", 0)]),
    ("ish", &[("ɪ", 0), ("ʃ", 0)]),
    ("able", &[("ə", 0), ("b", 0), ("əl", 0)]),
    ("ism", &[("ɪ", 0), ("z", 0), ("ə", 0), ("m", 0)]),
    ("ist", &[("ɪ", 0), ("s", 0), ("t", 0)]),
    ("wards", &[("w", 0), ("ɚ", 0), ("d", 0), ("z", 0)]),
    ("ward", &[("w", 0), ("ɚ", 0), ("d", 0)]),
    ("age", &[("ɪ", 0), ("dʒ", 0)]),
    ("ers", &[("ɚ", 0), ("z", 0)]),
    ("er", &[("ɚ", 0)]),
    ("est", &[("ɪ", 0), ("s", 0), ("t", 0)]),
    ("y", &[("i", 0)]),
];
// Prefix spelling -> (form before a stem stressed on its first syllable, form otherwise).
const PREFIXES: [(&str, &[Ph], &[Ph]); 12] = [
    ("under", &[("ʌ", 2), ("n", 0), ("d", 0), ("ɚ", 0)], &[("ʌ", 2), ("n", 0), ("d", 0), ("ɚ", 0)]),
    ("over", &[("oʊ", 2), ("v", 0), ("ɚ", 0)], &[("oʊ", 2), ("v", 0), ("ɚ", 0)]),
    ("anti", &[("æ", 2), ("n", 0), ("t", 0), ("i", 0)], &[("æ", 2), ("n", 0), ("t", 0), ("i", 0)]),
    ("semi", &[("s", 0), ("ɛ", 2), ("m", 0), ("i", 0)], &[("s", 0), ("ɛ", 2), ("m", 0), ("i", 0)]),
    ("non", &[("n", 0), ("ɑː", 2), ("n", 0)], &[("n", 0), ("ɑː", 2), ("n", 0)]),
    ("out", &[("aʊ", 2), ("t", 0)], &[("aʊ", 2), ("t", 0)]),
    ("dis", &[("d", 0), ("ɪ", 0), ("s", 0)], &[("d", 0), ("ɪ", 2), ("s", 0)]),
    ("mis", &[("m", 0), ("ɪ", 0), ("s", 0)], &[("m", 0), ("ɪ", 2), ("s", 0)]),
    ("un", &[("ʌ", 0), ("n", 0)], &[("ʌ", 2), ("n", 0)]),
    ("in", &[("ɪ", 0), ("n", 0)], &[("ɪ", 2), ("n", 0)]),
    ("re", &[("ɹ", 0), ("ᵻ", 0)], &[("ɹ", 0), ("iː", 2)]),
    ("pre", &[("p", 0), ("ɹ", 0), ("iː", 2)], &[("p", 0), ("ɹ", 0), ("iː", 2)]),
];
const BRITISH: [(&str, &str); 23] = [
    ("isation", "ization"), ("ising", "izing"), ("ised", "ized"), ("ises", "izes"), ("ise", "ize"), ("yse", "yze"),
    ("our", "or"), ("ourable", "orable"), ("ours", "ors"), ("oured", "ored"), ("ouring", "oring"),
    ("tre", "ter"), ("tres", "ters"), ("bre", "ber"), ("ence", "ense"), ("ogue", "og"), ("mme", "m"),
    ("lled", "led"), ("lling", "ling"), ("ller", "ler"), ("oul", "ol"), ("ae", "e"), ("enqui", "inqui"),
];

/// Stem + ending, adjusted at the seam.
pub fn join(stem: &[Ph], ending: &[Ph]) -> Phones {
    let mut stem: Phones = stem.to_vec();
    if stem.is_empty() || ending.is_empty() {
        stem.extend_from_slice(ending);
        return stem;
    }
    let (last, stress) = stem[stem.len() - 1];
    let (first, first_stress) = ending[0];
    let n = stem.len();
    if is_vowel(first) {
        if let Some(plain) = split_rhotic(last) {
            stem[n - 1] = (plain, stress);
            stem.push(("ɹ", 0));
        } else if RHOTIC.contains(&last) {
            stem.push(("ɹ", 0));
        } else if last == "t"
            && n > 1
            && (is_vowel(stem[n - 2].0) || stem[n - 2].0 == "ɹ")
            && first_stress == 0
            && !(first == "ə" && ending.len() > 1 && ending[1].0 == "n")
        {
            stem[n - 1] = ("ɾ", 0);
        } else if last == "i" {
            stem[n - 1] = if first != "ɪ" { ("ɪ", stress) } else { ("i", stress) };
        }
    } else if last == "i" && matches!(first, "n" | "l" | "f" | "m") {
        stem[n - 1] = ("ɪ", stress);
    } else if last == "n" && matches!(first, "k" | "ɡ") {
        stem[n - 1] = ("ŋ", 0);
    }
    stem.extend_from_slice(ending);
    stem
}

/// The regular -s ending. The stem must not be empty.
pub fn plural(stem: &[Ph]) -> Phones {
    let last = stem[stem.len() - 1].0;
    if SIBILANT.contains(&last) {
        join(stem, &[("ᵻ", 0), ("z", 0)])
    } else if VOICELESS.contains(&last) {
        join(stem, &[("s", 0)])
    } else {
        join(stem, &[("z", 0)])
    }
}

fn past(stem: &[Ph]) -> Phones {
    let last = stem[stem.len() - 1].0;
    if matches!(last, "t" | "d" | "ɾ") {
        return join(stem, &[("ᵻ", 0), ("d", 0)]);
    }
    if VOICELESS.contains(&last) || matches!(last, "s" | "ʃ" | "tʃ") {
        join(stem, &[("t", 0)])
    } else {
        join(stem, &[("d", 0)])
    }
}

fn adverb(stem: &[Ph]) -> Phones {
    let last = stem[stem.len() - 1].0;
    if matches!(last, "əl" | "l") { join(stem, &[("i", 0)]) } else { join(stem, &[("l", 0), ("i", 0)]) }
}

/// `(?:^|[^aeiou])[aeiou](?:[bcdfgklmnprstvz]|th)$`
fn short_stem(body: &str) -> bool {
    let b = body.as_bytes();
    let vowel = |c: u8| b"aeiou".contains(&c);
    let check = |tail: usize| -> bool {
        // `tail` letters close the stem; before them one vowel, before that no vowel
        if b.len() < tail + 1 || !vowel(b[b.len() - tail - 1]) {
            return false;
        }
        b.len() == tail + 1 || !vowel(b[b.len() - tail - 2])
    };
    (!b.is_empty() && b"bcdfgklmnprstvz".contains(&b[b.len() - 1]) && check(1)) || (body.ends_with("th") && check(2))
}

/// Spellings the stem may have had before an ending starting with a vowel.
fn stems(body: &str) -> Vec<String> {
    let chars: Vec<char> = body.chars().collect();
    let n = chars.len();
    let mut out = vec![body.to_string()];
    if n > 2 && chars[n - 1] == chars[n - 2] && consonant_letter(chars[n - 1]) {
        out.push(chars[..n - 1].iter().collect());
    }
    if n > 0 && (consonant_letter(chars[n - 1]) || chars[n - 1] == 'u') {
        let position = if short_stem(body) { 0 } else { out.len() };
        out.insert(position, format!("{body}e"));
    }
    if body.ends_with('i') && n > 2 {
        out.push(format!("{}y", chars[..n - 1].iter().collect::<String>()));
    }
    out
}

fn secondary(phones: &[Ph]) -> Phones {
    phones.iter().map(|&(p, s)| (p, if s == 1 { 2 } else { s })).collect()
}

pub type Lookup<'a> = &'a dyn Fn(&str) -> Option<Phones>;

fn nonempty(found: Option<Phones>) -> Option<Phones> {
    found.filter(|phones| !phones.is_empty())
}

/// Pronunciation of `word` from a shorter dictionary word, or `None`.
/// `lookup` gives the dictionary pronunciation; `part` does the same for a
/// word that may be half of a compound.
pub fn derive(word: &str, lookup: Lookup, part: Lookup, depth: usize) -> Option<Phones> {
    if !word.is_ascii() {
        return None;
    }
    let n = word.len();
    let stem = |spelling: &str| -> Option<Phones> {
        if spelling.len() < 2 {
            return None;
        }
        nonempty(lookup(spelling)).or_else(|| if depth < 2 { derive(spelling, lookup, part, depth + 1) } else { None })
    };
    // The first candidate with a stem decides, even if nothing can be built on it.
    let first = |candidates: &[String], build: &dyn Fn(&[Ph]) -> Option<Phones>| -> Option<Phones> {
        for spelling in candidates {
            if let Some(found) = nonempty(stem(spelling)) {
                return nonempty(build(&found));
            }
        }
        None
    };
    let some = |f: fn(&[Ph]) -> Phones| move |found: &[Ph]| Some(f(found));

    if word.ends_with("'s") {
        return first(&[word[..n - 2].to_string()], &some(plural));
    }
    if word.ends_with("s'") {
        return first(&[word[..n - 1].to_string()], &|found| Some(found.to_vec()));
    }
    if word.ends_with("ies") && n > 4 {
        if let Some(found) = first(&[format!("{}y", &word[..n - 3])], &some(plural)) {
            return Some(found);
        }
    }
    if word.ends_with("ves") && n > 4 {
        let build = |f: &[Ph]| -> Option<Phones> {
            if f[f.len() - 1].0 == "f" {
                let mut voiced = f[..f.len() - 1].to_vec();
                voiced.push(("v", 0));
                Some(join(&voiced, &[("z", 0)]))
            } else {
                None
            }
        };
        if let Some(found) = first(&[format!("{}f", &word[..n - 3]), format!("{}fe", &word[..n - 3])], &build) {
            return Some(found);
        }
    }
    if word.ends_with("es") && n > 3 {
        let candidates = if b"sxzh".contains(&word.as_bytes()[n - 3]) { vec![word[..n - 2].to_string()] } else { vec![] };
        if let Some(found) = first(&candidates, &some(plural)) {
            return Some(found);
        }
    }
    if word.ends_with('s') && !(word.ends_with("ss") || word.ends_with("us") || word.ends_with("is")) && n > 3 {
        if let Some(found) = first(&[word[..n - 1].to_string()], &some(plural)) {
            return Some(found);
        }
    }
    if word.ends_with("ied") && n > 4 {
        if let Some(found) = first(&[format!("{}y", &word[..n - 3])], &some(past)) {
            return Some(found);
        }
    }
    if word.ends_with("ed") && n > 3 {
        if let Some(found) = first(&stems(&word[..n - 2]), &some(past)) {
            return Some(found);
        }
    }
    if word.ends_with("ing") && n > 4 {
        if let Some(found) = first(&stems(&word[..n - 3]), &|f| Some(join(f, &[("ɪ", 0), ("ŋ", 0)]))) {
            return Some(found);
        }
    }
    if word.ends_with("ily") && n > 4 {
        if let Some(found) = first(&[format!("{}y", &word[..n - 3])], &some(adverb)) {
            return Some(found);
        }
    }
    if word.ends_with("ly") && n > 3 {
        let candidates = if word.ends_with("bly") || word.ends_with("ply") {
            vec![word[..n - 2].to_string(), format!("{}le", &word[..n - 2]), format!("{}e", &word[..n - 1])]
        } else {
            vec![word[..n - 2].to_string()]
        };
        if let Some(found) = first(&candidates, &some(adverb)) {
            return Some(found);
        }
    }
    for (spelling, phones) in ENDINGS {
        if word.ends_with(spelling) && n - spelling.len() > 2 {
            let body = &word[..n - spelling.len()];
            if spelling == "y" && !consonant_letter(body.chars().next_back().unwrap()) {
                continue;
            }
            let candidates = if is_vowel(phones[0].0) {
                stems(body)
            } else if body.ends_with('i') {
                vec![format!("{}y", &body[..body.len() - 1]), body.to_string()]
            } else {
                vec![body.to_string()]
            };
            if let Some(found) = first(&candidates, &|f| Some(join(f, phones))) {
                return Some(found);
            }
        }
    }
    for (spelling, plain, stressed) in PREFIXES {
        if word.starts_with(spelling) && n - spelling.len() > 2 {
            if let Some(found) = nonempty(stem(&word[spelling.len()..])) {
                let first_vowel = found.iter().find(|(p, _)| is_vowel(p)).map(|(_, s)| *s);
                let head = if first_vowel == Some(1) { plain } else { stressed };
                return Some(join(head, &found));
            }
        }
    }
    for (old, new) in BRITISH {
        if word.contains(old) {
            if let Some(found) = nonempty(stem(&word.replacen(old, new, 1))) {
                return Some(found);
            }
        }
    }
    if depth == 0 {
        // Two dictionary words of four letters or more: "sandpiper", "houseplant".
        let mut best: Option<(usize, Phones, Phones)> = None;
        for cut in 4..n.saturating_sub(3) {
            if let (Some(left), Some(right)) = (nonempty(part(&word[..cut])), nonempty(part(&word[cut..]))) {
                let score = cut.min(n - cut);
                if best.as_ref().is_none_or(|b| score > b.0) {
                    best = Some((score, left, right));
                }
            }
        }
        if let Some((_, left, right)) = best {
            return Some(join(&left, &secondary(&right)));
        }
    }
    None
}

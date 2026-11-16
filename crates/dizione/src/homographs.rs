//! Words spelled alike and pronounced differently: hand rules first, then a
//! small linear classifier over the neighbouring words.

use std::collections::{BTreeSet, HashMap, HashSet};

use serde_json::Value;

use crate::pyunicode as py;
use crate::tables;

const CLASSES: [(&str, &str); 13] = [
    ("TO", "to"),
    ("DET", "the a an this that these those every each no some any another such what which whose"),
    ("POSS", "my your his her its our their"),
    ("SUBJ", "i you he she it we they who"),
    ("OBJ", "me him us them"),
    ("MODAL", "will would shall should can could may might must 'll 'd won't can't cannot couldn't shouldn't wouldn't let's please"),
    ("DO", "do does did don't doesn't didn't"),
    ("BE", "am is are was were be been being isn't aren't wasn't weren't i'm you're we're they're he's she's it's that's there's"),
    ("HAVE", "have has had having haven't hasn't hadn't i've you've we've they've"),
    ("NEG", "not never"),
    ("PREP", "of in on at by for from with into onto over under about after before between through during without within \
              against among around across behind beside near upon toward towards as than like"),
    ("CONJ", "and or but nor so yet if because although though while when where unless until since whether"),
    ("ADV", "very too quite rather more most less least also just only even still always often sometimes usually again \
             now then here there soon already almost"),
];
const WINDOW: usize = 6;

struct Sense {
    id: String,
    role: String,
    phones: String,
}

type Weights = HashMap<String, f64>;

enum Model {
    NounVerb { bias: f64, weights: Weights },
    Senses { bias: Vec<f64>, weights: Vec<Weights> },
}

struct Entry {
    senses: Vec<Sense>,
    model: Model,
}

type Condition = (String, HashSet<String>);

pub struct Homographs {
    words: HashMap<String, Entry>,
    pooled_bias: f64,
    pooled_weights: Weights,
    plural_bases: HashSet<String>,
    rules: HashMap<String, Vec<(String, Vec<Condition>)>>,
    tags: HashMap<&'static str, &'static str>,
}

fn weights(value: &Value) -> Result<Weights, String> {
    let mut out = HashMap::new();
    for (name, weight) in value.as_object().ok_or("homographs.json: expected weights")? {
        out.insert(name.clone(), weight.as_f64().ok_or("homographs.json: expected a number")?);
    }
    Ok(out)
}

fn text(value: &Value, key: &str) -> Result<String, String> {
    value[key].as_str().map(String::from).ok_or_else(|| format!("homographs.json: missing {key}"))
}

impl Homographs {
    pub fn from_json(source: &str) -> Result<Self, String> {
        let data: Value = serde_json::from_str(source).map_err(|e| format!("homographs.json: {e}"))?;
        let mut words = HashMap::new();
        for (word, entry) in data["words"].as_object().ok_or("homographs.json: missing words")? {
            let mut senses = Vec::new();
            for sense in entry["senses"].as_array().ok_or("homographs.json: missing senses")? {
                senses.push(Sense { id: text(sense, "id")?, role: text(sense, "role")?, phones: text(sense, "phones")? });
            }
            let model = &entry["model"];
            let model = if model["kind"] == "noun-verb" {
                Model::NounVerb { bias: model["bias"].as_f64().ok_or("homographs.json: bias")?, weights: weights(&model["weights"])? }
            } else {
                Model::Senses {
                    bias: model["bias"].as_array().ok_or("homographs.json: bias")?.iter().map(|b| b.as_f64().unwrap_or(0.0)).collect(),
                    weights: model["weights"].as_array().ok_or("homographs.json: weights")?.iter().map(weights).collect::<Result<_, _>>()?,
                }
            };
            words.insert(word.clone(), Entry { senses, model });
        }
        let plural_bases = data["plural_bases"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
        let mut rules: HashMap<String, Vec<(String, Vec<Condition>)>> = HashMap::new();
        for (number, columns) in tables::rows(tables::HOMOGRAPH_RULES) {
            if columns.len() != 3 {
                return Err(format!("homograph_rules.txt line {number}: expected word, sense, conditions"));
            }
            let mut conditions = Vec::new();
            for part in columns[2].split(';') {
                let part = py::strip(part);
                let (name, list) = part.split_once(':').unwrap_or((part, ""));
                conditions.push((name.to_string(), py::split(list).map(String::from).collect()));
            }
            rules.entry(columns[0].to_string()).or_default().push((columns[1].to_string(), conditions));
        }
        let mut tags = HashMap::new();
        for (name, list) in CLASSES {
            for word in py::split(list) {
                tags.insert(word, name);
            }
        }
        Ok(Homographs {
            words,
            pooled_bias: data["pooled"]["bias"].as_f64().ok_or("homographs.json: pooled bias")?,
            pooled_weights: weights(&data["pooled"]["weights"])?,
            plural_bases,
            rules,
            tags,
        })
    }

    /// True for a homograph and for the regular -s form of a noun-or-verb one.
    pub fn covers(&self, word: &str) -> bool {
        self.words.contains_key(word) || (word.ends_with('s') && self.plural_bases.contains(&word[..word.len() - 1]))
    }

    /// Coarse grammatical class of a neighbouring token.
    fn tag(&self, token: &str) -> String {
        if token == "<s>" || token == "</s>" {
            return token.to_string();
        }
        let first = py::first(token).unwrap();
        if !py::is_alpha(first) {
            return if py::is_digit(first) { "NUM" } else { "PUNCT" }.to_string();
        }
        let lower = token.to_lowercase();
        if let Some(name) = self.tags.get(lower.as_str()) {
            return name.to_string();
        }
        let length = lower.chars().count();
        for (ending, name) in [("'s", "POS"), ("ly", "LY"), ("ing", "ING"), ("ed", "ED")] {
            if lower.ends_with(ending) && length > ending.len() + 1 {
                return name.to_string();
            }
        }
        if lower.ends_with('s') && !lower.ends_with("ss") && length > 3 {
            return "S".to_string();
        }
        if first.is_uppercase() { "CAP" } else { "W" }.to_string()
    }

    /// Names of the features that are on for `tokens[index]` (one sentence).
    fn features(&self, tokens: &[&str], index: usize) -> Vec<String> {
        let at = |offset: isize| -> &str {
            let position = index as isize + offset;
            if position >= 0 && (position as usize) < tokens.len() {
                tokens[position as usize]
            } else if offset < 0 {
                "<s>"
            } else {
                "</s>"
            }
        };
        let (p2, p1, n1, n2) = (at(-2), at(-1), at(1), at(2));
        let (tp2, tp1, tn1, tn2) = (self.tag(p2), self.tag(p1), self.tag(n1), self.tag(n2));
        let mut out = vec![
            format!("tp1={tp1}"), format!("tp2={tp2}"), format!("tn1={tn1}"), format!("tn2={tn2}"),
            format!("tp2p1={tp2}+{tp1}"), format!("tp1n1={tp1}+{tn1}"), format!("tn1n2={tn1}+{tn2}"),
        ];
        if index == 0 {
            out.push("first".into());
        } else if py::first(tokens[index]).unwrap().is_uppercase() {
            out.push("cap".into());
        }
        out.push(format!("p1={}", p1.to_lowercase()));
        out.push(format!("p2={}", p2.to_lowercase()));
        out.push(format!("n1={}", n1.to_lowercase()));
        out.push(format!("n2={}", n2.to_lowercase()));
        let target = tokens[index].to_lowercase();
        let from = index.saturating_sub(WINDOW);
        let to = (index + WINDOW + 1).min(tokens.len());
        let mut nearby: BTreeSet<String> =
            tokens[from..to].iter().filter(|t| py::is_alpha(py::first(t).unwrap())).map(|t| t.to_lowercase()).collect();
        nearby.remove(&target);
        out.extend(nearby.into_iter().map(|word| format!("w={word}")));
        out
    }

    /// The nearest token to the left that is not of a skipped class.
    fn left(&self, tokens: &[&str], index: usize, skip: &[&str]) -> String {
        let stop = index as isize - 6;
        let mut position = index as isize - 1;
        while position > stop.max(-1) {
            let token = tokens[position as usize];
            position -= 1;
            if skip.contains(&self.tag(token).as_str()) {
                continue;
            }
            let lower = token.to_lowercase();
            return if lower.ends_with("'ll") { "will".into() } else if lower.ends_with("'ve") { "have".into() } else { lower };
        }
        "<s>".into()
    }

    fn holds(&self, condition: &str, words: &HashSet<String>, tokens: &[&str], index: usize) -> bool {
        const BEFORE: [&str; 3] = ["ADV", "NEG", "LY"];
        const AUX: [&str; 7] = ["ADV", "NEG", "LY", "SUBJ", "OBJ", "DET", "POSS"];
        match condition {
            "first" => index == 0,
            "aux" => words.contains(&self.left(tokens, index, &AUX)),
            "before" => words.contains(&self.left(tokens, index, &BEFORE)),
            "next" => words.contains(&if index + 1 < tokens.len() { tokens[index + 1].to_lowercase() } else { "</s>".to_string() }),
            _ => {
                let from = index.saturating_sub(WINDOW);
                let to = (index + WINDOW + 1).min(tokens.len());
                tokens[from..to].iter().any(|token| words.contains(&token.to_lowercase()))
            }
        }
    }

    /// Score of every sense of `word` (higher is more likely).
    fn scores(&self, entry: &Entry, sentence: &[&str], position: usize) -> Vec<f64> {
        let names = self.features(sentence, position);
        match &entry.model {
            Model::NounVerb { bias, weights } => {
                let mut pooled = 0.0;
                for name in names.iter().filter(|n| n.starts_with('t') || *n == "first" || *n == "cap") {
                    pooled += self.pooled_weights.get(name).copied().unwrap_or(0.0);
                }
                let mut verb = self.pooled_bias + pooled;
                let mut own = 0.0;
                for name in &names {
                    own += weights.get(name).copied().unwrap_or(0.0);
                }
                verb += bias + own;
                entry.senses.iter().map(|sense| if sense.role == "v" { verb } else { -verb }).collect()
            }
            Model::Senses { bias, weights } => bias
                .iter()
                .zip(weights)
                .map(|(bias, weights)| {
                    let mut total = 0.0;
                    for name in &names {
                        total += weights.get(name).copied().unwrap_or(0.0);
                    }
                    bias + total
                })
                .collect(),
        }
    }

    /// The sense meant by `tokens[index]`.
    fn choose(&self, word: &str, tokens: &[&str], index: usize) -> &Sense {
        let entry = &self.words[word];
        let mut start = index;
        while start > 0 && !matches!(tokens[start - 1], "." | "?" | "!") {
            start -= 1;
        }
        let mut end = index;
        while end < tokens.len() && !matches!(tokens[end], "." | "?" | "!") {
            end += 1;
        }
        let sentence = &tokens[start..(end + 1).min(tokens.len())];
        let position = index - start;
        if let Some(rules) = self.rules.get(word) {
            for (sense, conditions) in rules {
                if conditions.iter().all(|(name, words)| self.holds(name, words, sentence, position)) {
                    if let Some(found) = entry.senses.iter().find(|s| s.id == *sense) {
                        return found;
                    }
                }
            }
        }
        let scores = self.scores(entry, sentence, position);
        let mut best = 0;
        for (k, score) in scores.iter().enumerate() {
            if *score > scores[best] {
                best = k;
            }
        }
        &entry.senses[best]
    }

    /// `(phones as text, needs the -s ending)` for `tokens[index]`.
    pub fn pronounce(&self, word: &str, tokens: &[&str], index: usize) -> (&str, bool) {
        if self.words.contains_key(word) {
            return (&self.choose(word, tokens, index).phones, false);
        }
        let base = &word[..word.len() - 1];
        let entry = &self.words[base];
        if let Some(adjective) = entry.senses.iter().find(|s| s.role == "n") {
            if adjective.id == "adj" {
                if let Some(verb) = entry.senses.iter().find(|s| s.role == "v") {
                    return (&verb.phones, true);
                }
            }
        }
        (&self.choose(base, tokens, index).phones, true)
    }
}

//! Which letters spell which phoneme: every phoneme of a pronunciation gets a
//! run of zero to four letters of the word, in order.

use std::collections::HashMap;

const MAX_RUN: usize = 4;
const UNSEEN: f64 = -14.0;

/// ARPABET symbol without its stress digit.
pub fn base(symbol: &str) -> &str {
    symbol.trim_end_matches(['0', '1', '2'])
}

pub struct Aligner {
    table: HashMap<String, HashMap<String, f64>>,
}

impl Aligner {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("align.json: {e}"))?;
        let mut table = HashMap::new();
        for (phone, runs) in value.as_object().ok_or("align.json: expected an object")? {
            let mut inner = HashMap::new();
            for (run, score) in runs.as_object().ok_or("align.json: expected nested objects")? {
                inner.insert(run.clone(), score.as_f64().ok_or("align.json: expected numbers")?);
            }
            table.insert(phone.clone(), inner);
        }
        Ok(Aligner { table })
    }

    /// Split `word` into one run of letters per phoneme (`phones` may carry
    /// stress digits). `None` when the word is too long for the pronunciation.
    pub fn align(&self, word: &str, phones: &[&str]) -> Option<Vec<String>> {
        let chars: Vec<char> = word.chars().collect();
        let (letters, count) = (chars.len(), phones.len());
        if count == 0 || letters > MAX_RUN * count {
            return None;
        }
        let slice = |from: usize, to: usize| -> String { chars[from..to].iter().collect() };
        let low = -1e18f64;
        let width = letters + 1;
        let mut best = vec![low; (count + 1) * width];
        let mut step = vec![0usize; (count + 1) * width];
        best[0] = 0.0;
        for j in 1..=count {
            let known = self.table.get(base(phones[j - 1]));
            for i in 0..=letters {
                for k in 0..=MAX_RUN.min(i) {
                    let previous = best[(j - 1) * width + i - k];
                    if previous > low {
                        let score = known
                            .and_then(|runs| runs.get(&slice(i - k, i)).copied())
                            .unwrap_or(UNSEEN - 2.0 * (k as f64 - 1.0).abs());
                        let value = previous + score;
                        if value > best[j * width + i] {
                            best[j * width + i] = value;
                            step[j * width + i] = k;
                        }
                    }
                }
            }
        }
        if best[count * width + letters] <= low {
            return None;
        }
        let mut runs = Vec::with_capacity(count);
        let mut i = letters;
        for j in (1..=count).rev() {
            let k = step[j * width + i];
            runs.push(slice(i - k, i));
            i -= k;
        }
        runs.reverse();
        Some(runs)
    }
}

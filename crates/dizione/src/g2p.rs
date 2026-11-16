//! The unknown-word model at run time: letters -> CMUdict phonemes. One
//! forward pass per word: letter embeddings (each letter given `slots`
//! positions), residual convolutions, a bidirectional GRU and a CTC output
//! layer read greedily.

use std::collections::HashMap;
use std::path::Path;

use unicode_normalization::UnicodeNormalization;

pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Arrays of an uncompressed `.npz` file (little-endian float32, C order).
fn read_npz(bytes: &[u8]) -> Result<HashMap<String, Array>, String> {
    let u16_at = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]) as usize;
    let u32_at = |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize;
    let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
    // End of central directory record.
    let end = (0..=bytes.len().saturating_sub(22)).rev().find(|&at| bytes[at..at + 4] == [0x50, 0x4b, 0x05, 0x06]).ok_or("g2p.npz: not a zip file")?;
    let (count, mut at) = (u16_at(end + 10), u32_at(end + 16));
    let mut arrays = HashMap::new();
    for _ in 0..count {
        if at + 46 > bytes.len() || bytes[at..at + 4] != [0x50, 0x4b, 0x01, 0x02] {
            return Err("g2p.npz: bad central directory".into());
        }
        let method = u16_at(at + 10);
        let (mut size, name_length, extra_length, comment_length) = (u32_at(at + 20), u16_at(at + 28), u16_at(at + 30), u16_at(at + 32));
        let mut local = u32_at(at + 42);
        let name = std::str::from_utf8(&bytes[at + 46..at + 46 + name_length]).map_err(|_| "g2p.npz: bad name")?.to_string();
        // Zip64 sizes and offsets.
        let mut extra = at + 46 + name_length;
        let extra_end = extra + extra_length;
        while extra + 4 <= extra_end {
            let (id, length) = (u16_at(extra), u16_at(extra + 2));
            if id == 1 {
                let mut field = extra + 4;
                if u32_at(at + 24) == 0xFFFF_FFFF { field += 8; }
                if size == 0xFFFF_FFFF { size = u64_at(field); field += 8; }
                if local == 0xFFFF_FFFF { local = u64_at(field); }
            }
            extra += 4 + length;
        }
        at += 46 + name_length + extra_length + comment_length;
        if method != 0 {
            return Err(format!("g2p.npz: {name} is compressed; save it with numpy.savez"));
        }
        let start = local + 30 + u16_at(local + 26) + u16_at(local + 28);
        let npy = bytes.get(start..start + size).ok_or("g2p.npz: truncated")?;
        arrays.insert(name.trim_end_matches(".npy").to_string(), read_npy(npy).map_err(|e| format!("g2p.npz: {name}: {e}"))?);
    }
    Ok(arrays)
}

fn read_npy(bytes: &[u8]) -> Result<Array, String> {
    if bytes.len() < 12 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not an npy array".into());
    }
    let (header_start, header_length) = if bytes[6] == 1 {
        (10, u16::from_le_bytes([bytes[8], bytes[9]]) as usize)
    } else {
        (12, u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize)
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_length]).map_err(|_| "bad header")?;
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        return Err("expected little-endian float32 in C order".into());
    }
    let open = header.find("'shape': (").ok_or("no shape")? + 10;
    let close = open + header[open..].find(')').ok_or("no shape")?;
    let shape: Vec<usize> = header[open..close].split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let data: Vec<f32> = bytes[header_start + header_length..].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    if data.len() != shape.iter().product::<usize>() {
        return Err("size does not match the shape".into());
    }
    Ok(Array { shape, data })
}

/// A matrix stored by columns, for products with a vector.
struct Columns {
    rows: usize,
    columns: usize,
    data: Vec<f32>,
}

impl Columns {
    /// From a row-major `[rows, columns]` array.
    fn new(array: &Array) -> Self {
        let (rows, columns) = (array.shape[0], array.shape[1]);
        let mut data = vec![0.0f32; rows * columns];
        for r in 0..rows {
            for c in 0..columns {
                data[c * rows + r] = array.data[r * columns + c];
            }
        }
        Columns { rows, columns, data }
    }

    /// `out += M v`, four columns at a time: every pass over `out` is a run
    /// of independent fused multiply-adds, which the compiler vectorizes.
    fn add_product(&self, v: &[f32], out: &mut [f32]) {
        let rows = self.rows;
        let out = &mut out[..rows];
        let mut column = 0;
        while column + 4 <= self.columns {
            let block = &self.data[column * rows..(column + 4) * rows];
            let (c0, rest) = block.split_at(rows);
            let (c1, rest) = rest.split_at(rows);
            let (c2, c3) = rest.split_at(rows);
            let (v0, v1, v2, v3) = (v[column], v[column + 1], v[column + 2], v[column + 3]);
            for i in 0..rows {
                out[i] = c3[i].mul_add(v3, c2[i].mul_add(v2, c1[i].mul_add(v1, c0[i].mul_add(v0, out[i]))));
            }
            column += 4;
        }
        while column < self.columns {
            let c0 = &self.data[column * rows..(column + 1) * rows];
            let v0 = v[column];
            for i in 0..rows {
                out[i] = c0[i].mul_add(v0, out[i]);
            }
            column += 1;
        }
    }
}

struct Gru {
    w_ih: Columns,
    w_hh: Columns,
    b_ih: Array,
    b_hh: Array,
}

pub struct Model {
    letters: HashMap<char, usize>,
    symbols: Vec<String>,
    slots: usize,
    embed: Array,
    slot: Array,
    convs: Vec<(Array, Array, Array, Array)>, // norm weight, norm bias, conv weight, conv bias
    layers: Vec<[Gru; 2]>,
    out_weight: Columns,
    out_bias: Array,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn gru(x: &[f32], frames: usize, g: &Gru, reverse: bool) -> Vec<f32> {
    let size = g.w_hh.columns;
    let width = g.w_ih.columns;
    let mut inputs = vec![0.0f32; frames * 3 * size];
    for t in 0..frames {
        let row = &mut inputs[t * 3 * size..(t + 1) * 3 * size];
        row.copy_from_slice(&g.b_ih.data);
        g.w_ih.add_product(&x[t * width..(t + 1) * width], row);
    }
    let mut state = vec![0.0f32; size];
    let mut recurrent = vec![0.0f32; 3 * size];
    let mut out = vec![0.0f32; frames * size];
    for step in 0..frames {
        let t = if reverse { frames - 1 - step } else { step };
        recurrent.copy_from_slice(&g.b_hh.data);
        g.w_hh.add_product(&state, &mut recurrent);
        let input = &inputs[t * 3 * size..(t + 1) * 3 * size];
        for k in 0..size {
            let reset = sigmoid(input[k] + recurrent[k]);
            let update = sigmoid(input[size + k] + recurrent[size + k]);
            let candidate = (input[2 * size + k] + reset * recurrent[2 * size + k]).tanh();
            state[k] = (1.0 - update) * candidate + update * state[k];
        }
        out[t * size..(t + 1) * size].copy_from_slice(&state);
    }
    out
}

impl Model {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let settings: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(directory.join("g2p.json")).map_err(|e| format!("g2p.json: {e}"))?)
            .map_err(|e| format!("g2p.json: {e}"))?;
        let mut arrays = read_npz(&std::fs::read(directory.join("g2p.npz")).map_err(|e| format!("g2p.npz: {e}"))?)?;
        let mut take = |name: String| arrays.remove(&name).ok_or(format!("g2p.npz: no array {name}"));
        let number = |key: &str| settings[key].as_u64().map(|n| n as usize).ok_or(format!("g2p.json: missing {key}"));
        let letters = settings["letters"].as_str().ok_or("g2p.json: missing letters")?.chars().enumerate().map(|(i, c)| (c, i + 1)).collect();
        let symbols = settings["symbols"].as_array().ok_or("g2p.json: missing symbols")?.iter().filter_map(|s| s.as_str().map(String::from)).collect();
        let mut convs = Vec::new();
        for block in 0..number("convs")? {
            convs.push((
                take(format!("norms__{block}__weight"))?,
                take(format!("norms__{block}__bias"))?,
                take(format!("convs__{block}__weight"))?,
                take(format!("convs__{block}__bias"))?,
            ));
        }
        let mut layers = Vec::new();
        for layer in 0..number("layers")? {
            let mut both = Vec::new();
            for suffix in ["", "_reverse"] {
                both.push(Gru {
                    w_ih: Columns::new(&take(format!("gru__weight_ih_l{layer}{suffix}"))?),
                    w_hh: Columns::new(&take(format!("gru__weight_hh_l{layer}{suffix}"))?),
                    b_ih: take(format!("gru__bias_ih_l{layer}{suffix}"))?,
                    b_hh: take(format!("gru__bias_hh_l{layer}{suffix}"))?,
                });
            }
            let reverse = both.pop().unwrap();
            layers.push([both.pop().unwrap(), reverse]);
        }
        Ok(Model {
            letters,
            symbols,
            slots: number("slots")?,
            embed: take("embed__weight".into())?,
            slot: take("slot".into())?,
            convs,
            layers,
            out_weight: Columns::new(&take("out__weight".into())?),
            out_bias: take("out__bias".into())?,
        })
    }

    /// The word reduced to the letters the model knows (accents removed).
    pub fn spellable(&self, word: &str) -> String {
        let plain: String = word.to_lowercase().nfkd().filter(|c| self.letters.contains_key(c)).collect();
        plain.trim_matches('\'').to_string()
    }

    /// CMUdict symbols with stress digits for `word`.
    pub fn predict(&self, word: &str) -> Vec<String> {
        let word = self.spellable(word);
        if word.is_empty() {
            return Vec::new();
        }
        let width = self.embed.shape[1];
        let ids: Vec<usize> = word.chars().map(|c| self.letters[&c]).collect();
        let frames = ids.len() * self.slots;
        let mut x = vec![0.0f32; frames * width];
        for (i, id) in ids.iter().enumerate() {
            for s in 0..self.slots {
                let row = &mut x[(i * self.slots + s) * width..(i * self.slots + s + 1) * width];
                for k in 0..width {
                    row[k] = self.embed.data[id * width + k] + self.slot.data[s * width + k];
                }
            }
        }
        for (norm_weight, norm_bias, weight, bias) in &self.convs {
            let (outputs, inputs, kernel) = (weight.shape[0], weight.shape[1], weight.shape[2]);
            let pad = kernel / 2;
            let mut normed = vec![0.0f32; frames * width];
            for t in 0..frames {
                let row = &x[t * width..(t + 1) * width];
                let mean = row.iter().sum::<f32>() / width as f32;
                let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
                let scale = (variance + 1e-5).sqrt();
                for k in 0..width {
                    normed[t * width + k] = (row[k] - mean) / scale * norm_weight.data[k] + norm_bias.data[k];
                }
            }
            for t in 0..frames {
                for o in 0..outputs {
                    let mut total = bias.data[o];
                    for k in 0..kernel {
                        let source = t + k;
                        if source < pad || source - pad >= frames {
                            continue;
                        }
                        let row = &normed[(source - pad) * width..(source - pad + 1) * width];
                        for i in 0..inputs {
                            total += row[i] * weight.data[(o * inputs + i) * kernel + k];
                        }
                    }
                    x[t * width + o] += total.max(0.0);
                }
            }
        }
        let mut width = width;
        for layer in &self.layers {
            let size = layer[0].w_hh.columns;
            let forward = gru(&x, frames, &layer[0], false);
            let backward = gru(&x, frames, &layer[1], true);
            let mut both = vec![0.0f32; frames * 2 * size];
            for t in 0..frames {
                both[t * 2 * size..t * 2 * size + size].copy_from_slice(&forward[t * size..(t + 1) * size]);
                both[t * 2 * size + size..(t + 1) * 2 * size].copy_from_slice(&backward[t * size..(t + 1) * size]);
            }
            x = both;
            width = 2 * size;
        }
        let labels = self.out_weight.rows;
        let mut logits = vec![0.0f32; labels];
        let (mut out, mut previous) = (Vec::new(), 0usize);
        for t in 0..frames {
            logits.copy_from_slice(&self.out_bias.data);
            self.out_weight.add_product(&x[t * width..(t + 1) * width], &mut logits);
            let mut label = 0;
            for k in 0..labels {
                if logits[k] > logits[label] {
                    label = k;
                }
            }
            if label != previous && label != 0 {
                out.push(self.symbols[label - 1].clone());
            }
            previous = label;
        }
        out
    }
}

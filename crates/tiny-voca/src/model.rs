//! The tiny-voca network for one utterance: text encoder, duration predictor,
//! frame expansion, prosody and acoustic transformers. Mirrors
//! `tiny_voca.model.TinyVoca.synthesize` of the PyTorch reference.

use std::path::Path;

use rayon::prelude::*;

use crate::ops::{self, LayerNorm, Linear};
use crate::vocoder::SourceFilter;
use crate::weights::Archive;

#[derive(Clone, Debug)]
pub struct Config {
    pub sample_rate: usize,
    pub hop: usize,
    pub f0_min: f32,
    pub f0_max: f32,
    pub speakers: Vec<String>,
    pub vocab_size: usize,
    pub channels: usize,
    pub heads: usize,
    pub feedforward: usize,
    pub kernel: usize,
    pub encoder_layers: usize,
    pub prosody_layers: usize,
    pub acoustic_layers: usize,
    pub predictor_channels: usize,
    pub synth_fft: usize,
    pub synth_hop: usize,
    pub noise_fft: usize,
    pub noise_hop: usize,
    pub cep_causal: usize,
    pub cep_anti: usize,
    pub noise_cep: usize,
    pub harmonic_limit: f64,
    pub max_tokens: usize,
    pub max_frames: usize,
    pub pitch_reference_hz: f32,
}

impl Config {
    fn from_json(text: &str) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("config: {e}"))?;
        let int = |key: &str| value[key].as_u64().map(|v| v as usize).ok_or_else(|| format!("config: missing {key}"));
        let float = |key: &str| value[key].as_f64().ok_or_else(|| format!("config: missing {key}"));
        let config = Self {
            sample_rate: int("sample_rate")?,
            hop: int("hop")?,
            f0_min: float("f0_min")? as f32,
            f0_max: float("f0_max")? as f32,
            speakers: value["speakers"].as_array().ok_or("config: missing speakers")?.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
            vocab_size: int("vocab_size")?,
            channels: int("channels")?,
            heads: int("heads")?,
            feedforward: int("feedforward")?,
            kernel: int("kernel")?,
            encoder_layers: int("encoder_layers")?,
            prosody_layers: int("prosody_layers")?,
            acoustic_layers: int("acoustic_layers")?,
            predictor_channels: int("predictor_channels")?,
            synth_fft: int("synth_fft")?,
            synth_hop: int("synth_hop")?,
            noise_fft: int("noise_fft")?,
            noise_hop: int("noise_hop")?,
            cep_causal: int("cep_causal")?,
            cep_anti: int("cep_anti")?,
            noise_cep: int("noise_cep")?,
            harmonic_limit: float("harmonic_limit")?,
            max_tokens: int("max_tokens")?,
            max_frames: int("max_frames")?,
            pitch_reference_hz: float("pitch_reference_hz")? as f32,
        };
        if config.vocab_size != SYMBOL_COUNT {
            return Err(format!("config: vocab_size {} does not match the {SYMBOL_COUNT}-symbol inventory", config.vocab_size));
        }
        if config.channels % config.heads != 0 || (config.channels / config.heads) % 2 != 0 || config.kernel % 2 == 0 {
            return Err("config: invalid transformer shape".into());
        }
        for (fft, hop) in [(config.synth_fft, config.synth_hop), (config.noise_fft, config.noise_hop)] {
            if fft != 4 * hop || config.hop % hop != 0 {
                return Err("config: synthesis windows need 4x overlap and a hop dividing the acoustic hop".into());
            }
        }
        Ok(config)
    }
}

// Inventory layout of `tiny_voca.text.SYMBOLS`: 5 specials, 12 punctuation
// marks, 26 consonants, 33 vowels.
const SYMBOL_COUNT: usize = 76;
const UNKNOWN: u32 = 4;
const FIRST_PHONE: u32 = 17;
const FIRST_VOWEL: u32 = 43;

fn is_phone(token: u32) -> bool {
    token >= FIRST_PHONE || token == UNKNOWN
}

fn is_vowel(token: u32) -> bool {
    token >= FIRST_VOWEL
}

struct Attention {
    qkv: Linear,
    out: Linear,
    heads: usize,
}

impl Attention {
    /// `x` is `[frames, channels]`; `table` holds cos and sin `[frames, width / 2]`.
    fn forward(&self, x: &[f32], table: &(Vec<f32>, Vec<f32>)) -> Vec<f32> {
        let channels = self.out.inputs;
        let frames = x.len() / channels;
        let width = channels / self.heads;
        let half = width / 2;
        let timer = crate::prof::scope("attn.qkv");
        let mut qkv = self.qkv.forward(x);
        drop(timer);
        let timer = crate::prof::scope("attn.rotary");
        // Rotary positions on queries and keys.
        let rotate = |first: usize, rows: &mut [f32]| {
            for (offset, row) in rows.chunks_exact_mut(3 * channels).enumerate() {
                let t = first + offset;
                let cos = &table.0[t * half..(t + 1) * half];
                let sin = &table.1[t * half..(t + 1) * half];
                for head in row[..2 * channels].chunks_exact_mut(width) {
                    for ((pair, c), s) in head.chunks_exact_mut(2).zip(cos).zip(sin) {
                        let (a, b) = (pair[0], pair[1]);
                        pair[0] = a * c - b * s;
                        pair[1] = a * s + b * c;
                    }
                }
            }
        };
        let stride = 3 * channels;
        let chunk = ops::row_chunk(frames, 4 * channels);
        if chunk >= frames {
            rotate(0, &mut qkv);
        } else {
            qkv.par_chunks_mut(chunk * stride).enumerate().for_each(|(index, rows)| rotate(index * chunk, rows));
        }
        let scale = 1.0 / (width as f32).sqrt();
        // Keys of each head as a contiguous [width, frames] matrix (K transposed).
        let mut keys = vec![0.0f32; channels * frames];
        let transpose = |first: usize, out: &mut [f32]| {
            for (offset, column) in out.chunks_exact_mut(frames).enumerate() {
                let i = channels + first + offset;
                column.iter_mut().zip(qkv.chunks_exact(stride)).for_each(|(value, row)| *value = row[i]);
            }
        };
        if chunk >= frames {
            transpose(0, &mut keys);
        } else {
            keys.par_chunks_mut(width * frames).enumerate().for_each(|(head, out)| transpose(head * width, out));
        }
        drop(timer);
        let (qkv, keys) = (&qkv, &keys);
        // Rows are processed in blocks small enough for the score matrix to stay
        // in cache between the product, the softmax and the weighted sum.
        let padded = frames.next_multiple_of(16);
        let block = (65536 / frames).clamp(4, 64);
        let mut merged = vec![0.0f32; frames * channels];
        let run = |index: usize, out: &mut [f32], (scores, scales): &mut (Vec<f32>, Vec<f32>)| {
            let rows = out.len() / channels;
            let first = index * block;
            scores.resize(rows * padded, 0.0);
            for h in 0..self.heads {
                unsafe {
                    let q = qkv.as_ptr().add(first * stride + h * width);
                    let k = keys.as_ptr().add(h * width * frames);
                    let v = qkv.as_ptr().add(2 * channels + h * width);
                    let timer = crate::prof::scope("attn.qk");
                    ops::sgemm(rows, width, frames, scale, q, stride, 1, k, frames, 1, 0.0, scores.as_mut_ptr(), padded);
                    drop(timer);
                    let timer = crate::prof::scope("attn.softmax");
                    ops::softmax_rows(scores, frames, padded, scales);
                    drop(timer);
                    let _timer = crate::prof::scope("attn.pv");
                    ops::sgemm(rows, frames, width, 1.0, scores.as_ptr(), padded, 1, v, stride, 1, 0.0, out.as_mut_ptr().add(h * width), channels);
                }
                for (row, scale) in out.chunks_exact_mut(channels).zip(scales.iter()) {
                    row[h * width..(h + 1) * width].iter_mut().for_each(|v| *v *= scale);
                }
            }
        };
        if rayon::current_num_threads() > 1 && frames > block {
            merged.par_chunks_mut(block * channels).enumerate().for_each_init(Default::default, |scratch, (index, out)| run(index, out, scratch));
        } else {
            let mut scratch = Default::default();
            merged.chunks_mut(block * channels).enumerate().for_each(|(index, out)| run(index, out, &mut scratch));
        }
        let _timer = crate::prof::scope("attn.out");
        self.out.forward(&merged)
    }
}

struct Block {
    norm1: LayerNorm,
    attention: Attention,
    norm2: LayerNorm,
    depthwise_weight: Vec<f32>, // [kernel, channels]
    depthwise_bias: Vec<f32>,
    kernel: usize,
    expand: Linear,
    project: Linear,
}

impl Block {
    fn forward(&self, x: &mut [f32], table: &(Vec<f32>, Vec<f32>)) {
        let timer = crate::prof::scope("norm");
        let normed = self.norm1.forward(x);
        drop(timer);
        let attended = self.attention.forward(&normed, table);
        x.iter_mut().zip(&attended).for_each(|(x, a)| *x += a);
        let timer = crate::prof::scope("norm");
        let normed = self.norm2.forward(x);
        drop(timer);
        let timer = crate::prof::scope("depthwise");
        let y = ops::depthwise(&normed, &self.depthwise_weight, &self.depthwise_bias, self.kernel);
        drop(timer);
        let timer = crate::prof::scope("expand");
        let mut y = self.expand.forward(&y);
        drop(timer);
        let timer = crate::prof::scope("gelu");
        ops::gelu(&mut y);
        drop(timer);
        let timer = crate::prof::scope("project");
        let y = self.project.forward(&y);
        drop(timer);
        x.iter_mut().zip(&y).for_each(|(x, y)| *x += y);
    }
}

struct Stack {
    blocks: Vec<Block>,
    norm: LayerNorm,
    width: usize,
}

impl Stack {
    fn forward(&self, mut x: Vec<f32>) -> Vec<f32> {
        let channels = self.norm.weight.len();
        let timer = crate::prof::scope("rotary_table");
        let table = rotary_table(x.len() / channels, self.width);
        drop(timer);
        for block in &self.blocks {
            block.forward(&mut x, &table);
        }
        self.norm.apply(&mut x);
        x
    }
}

fn rotary_table(size: usize, width: usize) -> (Vec<f32>, Vec<f32>) {
    let inverse: Vec<f32> = (0..width / 2).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / width as f32)).collect();
    let mut cos = Vec::with_capacity(size * inverse.len());
    let mut sin = Vec::with_capacity(size * inverse.len());
    for t in 0..size {
        for &frequency in &inverse {
            let angle = t as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

struct Predictor {
    conv1: Vec<Vec<f32>>,
    bias1: Vec<f32>,
    conv2: Vec<Vec<f32>>,
    bias2: Vec<f32>,
    norm1: LayerNorm,
    norm2: LayerNorm,
    out: Linear,
    inputs: usize,
}

impl Predictor {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = ops::conv1d(x, &self.conv1, &self.bias1, self.inputs);
        ops::gelu(&mut y);
        self.norm1.apply(&mut y);
        let mut y = ops::conv1d(&y, &self.conv2, &self.bias2, self.bias1.len());
        ops::gelu(&mut y);
        self.norm2.apply(&mut y);
        self.out.forward(&y)
    }
}

/// Options of free-running synthesis (the reference's defaults).
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Speaking rate, 0.25 to 4.
    pub speed: f32,
    /// Added to log F0 (0.058 is about a semitone).
    pub pitch_shift: f32,
    /// Scales pitch movement around the utterance mean, 0 to 3.
    pub pitch_range: f32,
    /// Seed of the noise source.
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self { speed: 1.0, pitch_shift: 0.0, pitch_range: 1.0, seed: 0 }
    }
}

pub struct Synthesis {
    pub audio: Vec<f32>,
    /// Frames (of `hop` samples) given to each token.
    pub durations: Vec<u32>,
    /// Per frame: log(F0 / reference), voicing logit, normalized log energy.
    pub prosody: Vec<[f32; 3]>,
    /// Oscillator frequency per frame in Hz.
    pub f0: Vec<f32>,
}

pub struct TinyVoca {
    pub config: Config,
    tokens: Vec<f32>,
    stress: Vec<f32>,
    voices: Vec<f32>,
    encoder: Stack,
    duration: Predictor,
    frame_voices: Vec<f32>,
    frame_position: Linear,
    prosody_stack: Stack,
    prosody_head: Linear,
    prosody_projection: Linear,
    acoustic_stack: Stack,
    vocoder: SourceFilter,
}

fn linear(archive: &mut Archive, name: &str, inputs: usize, outputs: usize) -> Result<Linear, String> {
    Ok(Linear { weight: archive.take(&format!("{name}.weight"), &[outputs, inputs])?, bias: archive.take(&format!("{name}.bias"), &[outputs])?, inputs, outputs })
}

fn layer_norm(archive: &mut Archive, name: &str, width: usize) -> Result<LayerNorm, String> {
    Ok(LayerNorm { weight: archive.take(&format!("{name}.weight"), &[width])?, bias: archive.take(&format!("{name}.bias"), &[width])? })
}

/// Split a `[outputs, inputs, kernel]` convolution weight into one
/// `[outputs, inputs]` matrix per kernel position.
fn conv_taps(weight: &[f32], outputs: usize, inputs: usize, kernel: usize) -> Vec<Vec<f32>> {
    (0..kernel).map(|tap| (0..outputs * inputs).map(|i| weight[i * kernel + tap]).collect()).collect()
}

fn stack(archive: &mut Archive, name: &str, c: &Config, layers: usize) -> Result<Stack, String> {
    let ch = c.channels;
    let mut blocks = Vec::with_capacity(layers);
    for index in 0..layers {
        let p = format!("{name}.blocks.{index}");
        let depthwise = archive.take(&format!("{p}.depthwise.weight"), &[ch, 1, c.kernel])?;
        blocks.push(Block {
            norm1: layer_norm(archive, &format!("{p}.norm1"), ch)?,
            attention: Attention { qkv: linear(archive, &format!("{p}.attention.qkv"), ch, 3 * ch)?, out: linear(archive, &format!("{p}.attention.out"), ch, ch)?, heads: c.heads },
            norm2: layer_norm(archive, &format!("{p}.norm2"), ch)?,
            depthwise_weight: (0..c.kernel * ch).map(|i| depthwise[(i % ch) * c.kernel + i / ch]).collect(),
            depthwise_bias: archive.take(&format!("{p}.depthwise.bias"), &[ch])?,
            kernel: c.kernel,
            expand: linear(archive, &format!("{p}.expand"), ch, c.feedforward)?,
            project: linear(archive, &format!("{p}.project"), c.feedforward, ch)?,
        });
    }
    Ok(Stack { blocks, norm: layer_norm(archive, &format!("{name}.norm"), ch)?, width: ch / c.heads })
}

impl TinyVoca {
    /// Load the safetensors file written by `scripts/convert_weights.py`.
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut archive = Archive::open(path)?;
        let config = Config::from_json(archive.metadata.get("config").ok_or("weights file has no config metadata")?)?;
        let c = &config;
        let (ch, p) = (c.channels, c.predictor_channels);
        let voices = c.speakers.len();
        let conv1 = archive.take("duration.conv1.weight", &[p, ch, 3])?;
        let conv2 = archive.take("duration.conv2.weight", &[p, p, 3])?;
        let harmonic_steps = c.hop / c.synth_hop;
        let noise_steps = c.hop / c.noise_hop;
        let model = Self {
            tokens: archive.take("tokens.weight", &[c.vocab_size, ch])?,
            stress: archive.take("stress.weight", &[3, ch])?,
            voices: archive.take("voices.weight", &[voices, ch])?,
            encoder: stack(&mut archive, "encoder", c, c.encoder_layers)?,
            duration: Predictor {
                conv1: conv_taps(&conv1, p, ch, 3),
                bias1: archive.take("duration.conv1.bias", &[p])?,
                conv2: conv_taps(&conv2, p, p, 3),
                bias2: archive.take("duration.conv2.bias", &[p])?,
                norm1: layer_norm(&mut archive, "duration.norm1", p)?,
                norm2: layer_norm(&mut archive, "duration.norm2", p)?,
                out: linear(&mut archive, "duration.out", p, 2)?,
                inputs: ch,
            },
            frame_voices: archive.take("frame_voices.weight", &[voices, ch])?,
            frame_position: linear(&mut archive, "frame_position", 3, ch)?,
            prosody_stack: stack(&mut archive, "prosody_stack", c, c.prosody_layers)?,
            prosody_head: linear(&mut archive, "prosody_head", ch, 3)?,
            prosody_projection: linear(&mut archive, "prosody_projection", 3, ch)?,
            acoustic_stack: stack(&mut archive, "acoustic_stack", c, c.acoustic_layers)?,
            vocoder: SourceFilter::new(
                c,
                linear(&mut archive, "vocoder.harmonic", ch, c.cep_causal + c.cep_anti + harmonic_steps)?,
                linear(&mut archive, "vocoder.noise", ch, c.noise_cep + noise_steps)?,
            ),
            config,
        };
        if let Some(name) = archive.tensors.keys().next() {
            return Err(format!("unexpected tensor {name} in the weights file"));
        }
        Ok(model)
    }

    pub fn voice_index(&self, name: &str) -> Option<usize> {
        self.config.speakers.iter().position(|speaker| speaker.eq_ignore_ascii_case(name))
    }

    /// Speak one utterance given phoneme ids and stress levels (edge tokens
    /// included). `noise` overrides the seeded noise source (parity tests).
    pub fn synthesize(&self, tokens: &[u32], stress: &[u8], voice: usize, options: &Options, noise: Option<&[f32]>) -> Result<Synthesis, String> {
        let _total = crate::prof::scope("TOTAL");
        let c = &self.config;
        let ch = c.channels;
        let Options { speed, pitch_shift, pitch_range, seed } = *options;
        if !speed.is_finite() || !(0.25..=4.0).contains(&speed) || !pitch_shift.is_finite() || !(0.0..=3.0).contains(&pitch_range) {
            return Err("speed must be between 0.25 and 4, pitch_shift finite and pitch_range between 0 and 3".into());
        }
        if tokens.is_empty() || tokens.len() != stress.len() || tokens.len() > c.max_tokens {
            return Err("invalid token length; split long text into sentences".into());
        }
        if voice >= c.speakers.len() || tokens.iter().any(|&t| t as usize >= c.vocab_size) || stress.iter().any(|&s| s > 2) {
            return Err("token, stress or voice index out of range".into());
        }
        // Text encoder.
        let voice_row = &self.voices[voice * ch..(voice + 1) * ch];
        let mut x = vec![0.0f32; tokens.len() * ch];
        for ((row, &token), &level) in x.chunks_exact_mut(ch).zip(tokens).zip(stress) {
            let token = &self.tokens[token as usize * ch..(token as usize + 1) * ch];
            let level = &self.stress[level as usize * ch..(level as usize + 1) * ch];
            for i in 0..ch {
                row[i] = token[i] + level[i] + voice_row[i];
            }
        }
        let hidden = self.encoder.forward(x);
        // Durations: every phoneme lasts at least a frame; a pause-capable token
        // lasts its predicted length when the presence logit says so, else nothing.
        let output = self.duration.forward(&hidden);
        let limit = 401f32.ln();
        let durations: Vec<u32> = output.chunks_exact(2).zip(tokens).map(|(out, &token)| {
            let frames = ((out[0].clamp(0.0, limit).exp_m1() / speed).round_ties_even() as u32).max(1);
            if is_phone(token) || out[1] > 0.0 { frames } else { 0 }
        }).collect();
        let total: usize = durations.iter().map(|&d| d as usize).sum();
        if total < 1 || total > c.max_frames {
            return Err("utterance has no frames or exceeds max_frames; split the input".into());
        }
        // Repeat token states by duration and add each frame's place in its token.
        let frame_voice = &self.frame_voices[voice * ch..(voice + 1) * ch];
        let mut features = Vec::with_capacity(total * 3);
        let mut owner = Vec::with_capacity(total);
        for (index, &duration) in durations.iter().enumerate() {
            for offset in 0..duration {
                let relative = (offset as f32 + 0.5) / duration as f32;
                features.extend_from_slice(&[relative, 1.0 - relative, (duration as f32).ln_1p() / 4.0]);
                owner.push(index);
            }
        }
        let mut frames = self.frame_position.forward(&features);
        for (row, &index) in frames.chunks_exact_mut(ch).zip(&owner) {
            let state = &hidden[index * ch..(index + 1) * ch];
            for i in 0..ch {
                row[i] += state[i] + frame_voice[i];
            }
        }
        // Pitch, voicing, energy.
        let prosody_hidden = self.prosody_stack.forward(frames);
        let raw = self.prosody_head.forward(&prosody_hidden);
        let mut prosody: Vec<[f32; 3]> = raw.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        // A vowel is voiced by definition: keep its frames voiced.
        for (frame, &index) in prosody.iter_mut().zip(&owner) {
            if is_vowel(tokens[index]) {
                frame[1] = frame[1].max(2.0);
            }
        }
        if pitch_range != 1.0 {
            let (sum, count) = prosody.iter().filter(|p| p[1] > 0.0).fold((0.0f32, 0.0f32), |(sum, count), p| (sum + p[0], count + 1.0));
            let mean = sum / count.max(1.0);
            prosody.iter_mut().for_each(|p| p[0] = mean + (p[0] - mean) * pitch_range);
        }
        let condition: Vec<f32> = prosody.iter().flat_map(|p| [p[0].clamp(-2.0, 1.5), (p[1] > 0.0) as u8 as f32, p[2].clamp(-4.5, 3.0)]).collect();
        let mut decoder_input = self.prosody_projection.forward(&condition);
        decoder_input.iter_mut().zip(&prosody_hidden).for_each(|(x, h)| *x += h);
        let acoustic = self.acoustic_stack.forward(decoder_input);
        let f0: Vec<f32> = prosody.iter().map(|p| (c.pitch_reference_hz * (p[0] + pitch_shift).exp()).clamp(c.f0_min, c.f0_max)).collect();
        let samples = total * c.hop;
        let drawn;
        let noise = match noise {
            Some(noise) if noise.len() == samples => noise,
            Some(_) => return Err("noise length does not match the utterance".into()),
            None => {
                let _timer = crate::prof::scope("noise.randn");
                drawn = crate::rng::randn(samples, seed);
                &drawn
            }
        };
        let timer = crate::prof::scope("vocoder.total");
        let audio = self.vocoder.forward(&acoustic, &f0, noise);
        drop(timer);
        if audio.iter().any(|v| !v.is_finite()) {
            return Err("model produced non-finite audio".into());
        }
        Ok(Synthesis { audio, durations, prosody, f0 })
    }
}

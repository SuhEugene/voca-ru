//! Source-filter waveform head: a band-limited pulse train at the predicted
//! pitch and a noise signal, each shaped by a time-varying filter whose
//! cepstrum the network predicts, applied in the short-time Fourier domain.

use std::sync::Arc;

use rayon::prelude::*;
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::fastmath::{exp, round64, sin_cos_turns};
use crate::model::Config;
use crate::ops::Linear;

struct Branch {
    fft: usize,
    hop: usize,
    window: Vec<f32>,
    /// 1 / (summed squared window) for the first hop of the signal, where one
    /// overlapping frame is missing, and for every later hop.
    first_gain: Vec<f32>,
    gain: Vec<f32>,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
}

/// Per-thread buffers of one branch.
struct Scratch {
    buffer: Vec<f32>,
    spectrum: Vec<Complex32>,
    real: Vec<f32>,
    imaginary: Vec<f32>,
    fft: Vec<Complex32>,
}

impl Branch {
    fn new(fft: usize, hop: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        // Periodic Hann window, as torch.hann_window.
        let window: Vec<f32> = (0..fft).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / fft as f64).cos()) as f32).collect();
        let full: Vec<f32> = (0..hop).map(|r| (0..4).map(|q| window[r + q * hop] * window[r + q * hop]).sum()).collect();
        Self {
            fft,
            hop,
            first_gain: (0..hop).map(|r| 1.0 / (full[r] - window[r + 3 * hop] * window[r + 3 * hop])).collect(),
            gain: full.iter().map(|v| 1.0 / v).collect(),
            window,
            forward: planner.plan_fft_forward(fft),
            inverse: planner.plan_fft_inverse(fft),
        }
    }

    fn bins(&self) -> usize {
        self.fft / 2 + 1
    }

    fn scratch(&self) -> Scratch {
        let bins = self.bins();
        Scratch {
            buffer: vec![0.0; self.fft],
            spectrum: vec![Complex32::default(); bins],
            real: vec![0.0; bins],
            imaginary: vec![0.0; bins],
            fft: vec![Complex32::default(); self.forward.get_scratch_len().max(self.inverse.get_scratch_len())],
        }
    }

    /// Spectra of per-frame cepstra: `fill(frame, buffer)` writes the cepstrum
    /// into the zeroed buffer; `store(frame, spectrum)` receives its transform.
    fn cepstra<T: Send>(&self, frames: usize, fill: impl Fn(usize, &mut [f32]) + Sync, store: impl Fn(&[Complex32], &mut [T]) + Sync, out: &mut [T]) {
        let width = out.len() / frames;
        let chunk = chunk_size(frames, 16);
        out.par_chunks_mut(chunk * width).enumerate().for_each(|(index, out)| {
            let mut scratch = self.scratch();
            for (offset, row) in out.chunks_exact_mut(width).enumerate() {
                scratch.buffer.fill(0.0);
                fill(index * chunk + offset, &mut scratch.buffer);
                self.forward.process_with_scratch(&mut scratch.buffer, &mut scratch.spectrum, &mut scratch.fft).unwrap();
                store(&scratch.spectrum, row);
            }
        });
    }

    /// Filter `source` with one complex response per sub-frame: `response(j,
    /// real, imaginary)` writes sub-frame j's response. Equals torch.stft /
    /// multiply / torch.istft with half a window of zeros appended to the
    /// source, cut back to the source's length.
    fn shape(&self, source: &[f32], response: impl Fn(usize, &mut [f32], &mut [f32]) + Sync) -> Vec<f32> {
        let (fft, hop) = (self.fft, self.hop);
        let half = fft / 2;
        let samples = source.len();
        let frames = samples / hop + 3;
        let bins = self.bins();
        let scale = 1.0 / fft as f32;
        let chunk = chunk_size(frames, 32);
        // Each task overlap-adds its frames into a local stretch of signal that
        // starts `half` samples before its first frame's centre.
        let parts: Vec<Vec<f32>> = (0..frames.div_ceil(chunk)).into_par_iter().map(|index| {
            let first = index * chunk;
            let count = chunk.min(frames - first);
            let mut local = vec![0.0f32; (count - 1) * hop + fft];
            let mut scratch = self.scratch();
            for offset in 0..count {
                let j = first + offset;
                // Frame j covers source samples [j * hop - half, j * hop + half).
                let start = j * hop;
                if start >= half && start + half <= samples {
                    let input = &source[start - half..start + half];
                    scratch.buffer.iter_mut().zip(input).zip(&self.window).for_each(|((value, input), window)| *value = input * window);
                } else {
                    for (n, (value, window)) in scratch.buffer.iter_mut().zip(&self.window).enumerate() {
                        *value = match (start + n).checked_sub(half) {
                            Some(position) if position < samples => source[position] * window,
                            _ => 0.0,
                        };
                    }
                }
                self.forward.process_with_scratch(&mut scratch.buffer, &mut scratch.spectrum, &mut scratch.fft).unwrap();
                response(j, &mut scratch.real, &mut scratch.imaginary);
                for ((value, real), imaginary) in scratch.spectrum.iter_mut().zip(&scratch.real).zip(&scratch.imaginary) {
                    *value = Complex32::new(value.re * real - value.im * imaginary, value.re * imaginary + value.im * real);
                }
                scratch.spectrum[0].im = 0.0;
                scratch.spectrum[bins - 1].im = 0.0;
                self.inverse.process_with_scratch(&mut scratch.spectrum, &mut scratch.buffer, &mut scratch.fft).unwrap();
                let target = &mut local[offset * hop..offset * hop + fft];
                target.iter_mut().zip(&scratch.buffer).zip(&self.window).for_each(|((value, piece), window)| *value += piece * window * scale);
            }
            local
        }).collect();
        let mut wave = vec![0.0f32; samples];
        for (index, local) in parts.iter().enumerate() {
            // local[n] belongs to source sample index * chunk * hop - half + n.
            let origin = index * chunk * hop;
            let skip = half.saturating_sub(origin);
            let begin = origin + skip - half;
            if begin >= samples {
                break;
            }
            let count = (local.len() - skip).min(samples - begin);
            wave[begin..begin + count].iter_mut().zip(&local[skip..skip + count]).for_each(|(value, local)| *value += local);
        }
        // Normalize by the summed squared window.
        for (index, block) in wave.chunks_mut(hop).enumerate() {
            let gain = if index == 0 { &self.first_gain } else { &self.gain };
            block.iter_mut().zip(gain).for_each(|(value, gain)| *value *= gain);
        }
        wave
    }
}

/// Items per parallel task: an even split over the threads, at least `minimum`.
fn chunk_size(items: usize, minimum: usize) -> usize {
    let threads = rayon::current_num_threads();
    if threads > 1 { items.div_ceil(threads).max(minimum) } else { items.max(1) }
}

pub struct SourceFilter {
    harmonic: Linear,
    noise: Linear,
    hop: usize,
    sample_rate: f64,
    harmonic_limit: f64,
    cep_causal: usize,
    cep_anti: usize,
    noise_cep: usize,
    harmonic_steps: usize,
    noise_steps: usize,
    causal_lifter: Vec<f32>,
    anti_lifter: Vec<f32>,
    noise_lifter: Vec<f32>,
    synth: Branch,
    hiss: Branch,
}

fn lifter(size: usize, first: usize) -> Vec<f32> {
    let mut scale: Vec<f32> = (first..first + size).map(|i| 1.0 / (1.0 + i as f32 / 4.0)).collect();
    if first == 0 {
        scale[0] = 2.0; // coefficient 0 is the log gain and needs range
    }
    scale
}

impl SourceFilter {
    pub fn new(c: &Config, harmonic: Linear, noise: Linear) -> Self {
        Self {
            harmonic,
            noise,
            hop: c.hop,
            sample_rate: c.sample_rate as f64,
            harmonic_limit: c.harmonic_limit,
            cep_causal: c.cep_causal,
            cep_anti: c.cep_anti,
            noise_cep: c.noise_cep,
            harmonic_steps: c.hop / c.synth_hop,
            noise_steps: c.hop / c.noise_hop,
            causal_lifter: lifter(c.cep_causal, 0),
            anti_lifter: lifter(c.cep_anti, 1),
            noise_lifter: lifter(c.noise_cep, 0),
            synth: Branch::new(c.synth_fft, c.synth_hop),
            hiss: Branch::new(c.noise_fft, c.noise_hop),
        }
    }

    /// `hidden` is `[frames, channels]`, `f0` in Hz per frame, `noise` one
    /// sample per output sample. Returns `frames * hop` samples.
    pub fn forward(&self, hidden: &[f32], f0: &[f32], noise: &[f32]) -> Vec<f32> {
        let frames = f0.len();
        let split = self.cep_causal + self.cep_anti;
        const TURNS: f32 = 1.0 / (2.0 * std::f32::consts::PI);
        let harmonic = || {
            // Envelope interpolated between frames, one gain per sub-frame. The
            // transform is linear, so the spectrum of an interpolated cepstrum is
            // the mean of the two frames' spectra and the gain adds to its real part.
            let h = self.harmonic.forward(hidden);
            let width = split + self.harmonic_steps;
            let steps = self.harmonic_steps;
            let bins = self.synth.bins();
            let timer = crate::prof::scope("voc.harmonic_cepstra");
            let mut spectra = vec![0.0f32; frames * 2 * bins]; // per frame: real parts, then imaginary parts in turns
            self.synth.cepstra(frames, |frame, buffer| {
                let row = &h[frame * width..][..width];
                for i in 0..self.cep_causal {
                    buffer[i] = row[i] * self.causal_lifter[i];
                }
                let last = buffer.len() - 1;
                for i in 0..self.cep_anti {
                    buffer[last - i] = row[self.cep_causal + i] * self.anti_lifter[i];
                }
            }, |spectrum, row| {
                let (real, imaginary) = row.split_at_mut(bins);
                for ((value, real), imaginary) in spectrum.iter().zip(real).zip(imaginary) {
                    *real = value.re;
                    *imaginary = value.im * TURNS;
                }
            }, &mut spectra);
            drop(timer);
            let timer = crate::prof::scope("voc.pulse_train");
            let pulses = pulse_train(f0, self.sample_rate, self.hop, self.harmonic_limit);
            drop(timer);
            let _timer = crate::prof::scope("voc.harmonic_shape");
            self.synth.shape(&pulses, |j, real, imaginary| {
                let sub = j.min(steps * frames - 1);
                let gain = h[(sub / steps) * width + split + sub % steps];
                let (here_re, here_im) = spectra[(sub / steps) * 2 * bins..][..2 * bins].split_at(bins);
                let (next_re, next_im) = spectra[((sub + 1) / steps).min(frames - 1) * 2 * bins..][..2 * bins].split_at(bins);
                for i in 0..bins {
                    let magnitude = exp((0.5 * (here_re[i] + next_re[i]) + gain).clamp(-14.0, 6.0));
                    let (sin, cos) = sin_cos_turns(0.5 * (here_im[i] + next_im[i]));
                    real[i] = magnitude * cos;
                    imaginary[i] = magnitude * sin;
                }
            })
        };
        let hiss = || {
            // One envelope per frame, several gains per frame for sharp consonants:
            // the phase is shared by a frame's sub-frames, only the level changes.
            let n = self.noise.forward(hidden);
            let width = self.noise_cep + self.noise_steps;
            let steps = self.noise_steps;
            let bins = self.hiss.bins();
            let timer = crate::prof::scope("voc.noise_cepstra");
            let mut spectra = vec![0.0f32; frames * 3 * bins]; // per frame: log magnitude, cos, sin
            self.hiss.cepstra(frames, |frame, buffer| {
                let row = &n[frame * width..][..width];
                for i in 0..self.noise_cep {
                    buffer[i] = row[i] * self.noise_lifter[i];
                }
            }, |spectrum, row| {
                let (level, rest) = row.split_at_mut(bins);
                let (cos, sin) = rest.split_at_mut(bins);
                for i in 0..bins {
                    level[i] = spectrum[i].re;
                    (sin[i], cos[i]) = sin_cos_turns(spectrum[i].im * TURNS);
                }
            }, &mut spectra);
            drop(timer);
            let _timer = crate::prof::scope("voc.noise_shape");
            self.hiss.shape(noise, |j, real, imaginary| {
                let sub = j.min(steps * frames - 1);
                let gain = n[(sub / steps) * width + self.noise_cep + sub % steps];
                let row = &spectra[(sub / steps) * 3 * bins..][..3 * bins];
                for i in 0..bins {
                    let magnitude = exp((row[i] + gain).clamp(-14.0, 6.0));
                    real[i] = magnitude * row[bins + i];
                    imaginary[i] = magnitude * row[2 * bins + i];
                }
            })
        };
        let (mut wave, hiss) = rayon::join(harmonic, hiss);
        wave.iter_mut().zip(&hiss).for_each(|(value, hiss)| *value += hiss);
        wave
    }
}

/// Band-limited pulse train with unit power for per-frame `f0` in Hz.
///
/// Frame values sit at samples `i * hop` and are interpolated in the log
/// domain. The sum of all harmonics below `limit * sample_rate` has the closed
/// form sin((K+1/2)x) / (2 sin(x/2)) - 1/2; the top harmonic fades in and out
/// so the signal stays continuous while the pitch glides. The phase is
/// accumulated in double precision; the trigonometry runs in single precision
/// on phases already reduced to one turn.
pub fn pulse_train(f0: &[f32], sample_rate: f64, hop: usize, limit: f64) -> Vec<f32> {
    let frames = f0.len();
    let mut knots: Vec<f64> = f0.iter().map(|&f| (f as f64).max(1.0).ln()).collect();
    knots.push(knots[frames - 1]);
    let mut wave = vec![0.0f32; frames * hop];
    let mut cycles = 0.0f64; // phase in turns, kept in [0, 1)
    let mut frequency = vec![0.0f64; hop];
    let mut phase = vec![0.0f64; hop];
    let mut half = vec![0.0f32; hop];
    let mut top = vec![0.0f32; hop];
    let mut harmonics = vec![0.0f32; hop];
    let ceiling = (limit * sample_rate) as f32;
    for (frame, out) in wave.chunks_exact_mut(hop).enumerate() {
        // Log-linear glide between the frame's knots: a geometric sequence.
        let mut value = knots[frame].exp();
        let ratio = ((knots[frame + 1] - knots[frame]) / hop as f64).exp();
        for step in 0..hop {
            frequency[step] = value;
            value *= ratio;
            cycles += frequency[step] / sample_rate;
            cycles -= cycles.floor();
            // Centred on the pulse, where the closed form needs the precision.
            phase[step] = if cycles >= 0.5 { cycles - 1.0 } else { cycles };
        }
        // theta / 2 and (K + 1/2) theta in turns, reduced in double precision.
        for step in 0..hop {
            let count = (ceiling / frequency[step] as f32).max(1.0);
            let whole = count.floor() as f64;
            let turns = (whole + 0.5) * phase[step];
            harmonics[step] = count;
            half[step] = (0.5 * phase[step]) as f32;
            top[step] = (turns - round64(turns)) as f32;
        }
        for step in 0..hop {
            let count = harmonics[step];
            let whole = count.floor();
            let (sin_half, cos_half) = sin_cos_turns(half[step]);
            let (sin_top, cos_top) = sin_cos_turns(top[step]);
            let denominator = 2.0 * sin_half;
            let tiny = denominator.abs() < 1e-7;
            let core = if tiny { whole } else { sin_top / if tiny { 1.0 } else { denominator } - 0.5 };
            // cos((K + 1) theta) = cos((K + 1/2) theta + theta / 2)
            let fade = cos_top * cos_half - sin_top * sin_half;
            out[step] = (core + (count - whole) * fade) * (2.0 / count).sqrt();
        }
    }
    wave
}

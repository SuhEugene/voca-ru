//! Compare the Rust model with reference dumps of the PyTorch model.
//!
//!     parity weights/tiny-voca.safetensors out/reference
//!
//! Each case file (from scripts/dump_reference.py) holds the inputs, the exact
//! noise PyTorch drew, and PyTorch's durations, prosody, F0 and waveform.

use std::path::Path;

use tiny_voca::weights::Archive;
use tiny_voca::{Options, TinyVoca};

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let signal: f64 = reference.iter().map(|&v| (v as f64).powi(2)).sum();
    let error: f64 = reference.iter().zip(test).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
    10.0 * (signal / error.max(1e-30)).log10()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: parity WEIGHTS REFERENCE_DIR".into());
    }
    let model = TinyVoca::load(Path::new(&args[1]))?;
    let mut paths: Vec<_> = std::fs::read_dir(&args[2]).map_err(|e| e.to_string())?.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors")).collect();
    paths.sort();
    let mut failures = 0;
    println!("{:<8} {:>6} {:>7} {:>9} {:>10} {:>10} {:>10} {:>9} {:>9}", "case", "tokens", "frames", "durations", "prosody", "f0 (Hz)", "wave", "SNR dB", "own-rng");
    for path in &paths {
        let case = Archive::open(path)?;
        let get = |name: &str| case.tensors.get(name).ok_or_else(|| format!("{name} missing"));
        let meta = |name: &str| case.metadata.get(name).cloned().unwrap_or_default();
        let tokens: Vec<u32> = get("tokens")?.i32.iter().map(|&v| v as u32).collect();
        let stress: Vec<u8> = get("stress")?.i32.iter().map(|&v| v as u8).collect();
        let options = Options { speed: meta("speed").parse().unwrap(), pitch_shift: meta("pitch_shift").parse().unwrap(),
                                pitch_range: meta("pitch_range").parse().unwrap(), seed: meta("seed").parse().unwrap() };
        let voice: usize = meta("voice").parse().unwrap();
        let result = model.synthesize(&tokens, &stress, voice, &options, Some(&get("noise")?.f32))?;
        let durations_equal = result.durations.iter().map(|&d| d as i32).eq(get("durations")?.i32.iter().copied());
        let name = path.file_stem().unwrap().to_string_lossy();
        if !durations_equal {
            failures += 1;
            println!("{name:<8} {:>6} durations differ", tokens.len());
            continue;
        }
        let prosody: Vec<f32> = result.prosody.iter().flatten().copied().collect();
        let wave = &get("wave")?.f32;
        let wave_diff = max_diff(&result.audio, wave);
        let snr = snr_db(wave, &result.audio);
        // Same seed, noise drawn by the Rust generator instead of handed over.
        let seeded = model.synthesize(&tokens, &stress, voice, &options, None)?;
        let seeded_diff = max_diff(&seeded.audio, wave);
        if snr < 45.0 || seeded_diff > 10.0 * wave_diff.max(1e-4) {
            failures += 1;
        }
        println!("{name:<8} {:>6} {:>7} {:>9} {:>10.2e} {:>10.2e} {:>10.2e} {:>9.1} {:>9.2e}", tokens.len(), result.f0.len(), "equal",
                 max_diff(&prosody, &get("prosody")?.f32), max_diff(&result.f0, &get("f0")?.f32), wave_diff, snr, seeded_diff);
    }
    println!("{} cases, {failures} failed (backend: {})", paths.len(), tiny_voca::ops::backend());
    if failures > 0 { Err("parity check failed".into()) } else { Ok(()) }
}

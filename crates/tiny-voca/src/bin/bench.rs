//! Time the model on the reference cases (same token inputs as the PyTorch
//! and ONNX Runtime benchmarks in scripts/bench_python.py).
//!
//!     bench WEIGHTS REFERENCE_DIR [--threads N] [--runs N] [--json]

use std::path::Path;
use std::time::Instant;

use tiny_voca::weights::Archive;
use tiny_voca::{Options, TinyVoca};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<usize>().ok());
    let threads = flag("--threads").unwrap_or(1);
    let runs = flag("--runs").unwrap_or(20);
    let json = args.iter().any(|a| a == "--json");
    if args.len() < 3 {
        return Err("usage: bench WEIGHTS REFERENCE_DIR [--threads N] [--runs N] [--json]".into());
    }
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().map_err(|e| e.to_string())?;
    let started = Instant::now();
    let model = TinyVoca::load(Path::new(&args[1]))?;
    let load_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut paths: Vec<_> = std::fs::read_dir(&args[2]).map_err(|e| e.to_string())?.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors")).collect();
    paths.sort();
    let only = args.iter().position(|a| a == "--only").and_then(|i| args.get(i + 1)).cloned();
    let profile = std::env::var_os("TINY_VOCA_PROFILE").is_some();
    let mut rows = Vec::new();
    for path in &paths {
        let case = Archive::open(path)?;
        if case.metadata.get("speed").map(String::as_str) != Some("1.0") {
            continue;
        }
        if only.as_ref().is_some_and(|only| !path.to_string_lossy().contains(only.as_str())) {
            continue;
        }
        let tokens: Vec<u32> = case.tensors["tokens"].i32.iter().map(|&v| v as u32).collect();
        let stress: Vec<u8> = case.tensors["stress"].i32.iter().map(|&v| v as u8).collect();
        let voice: usize = case.metadata["voice"].parse().unwrap();
        let options = Options::default();
        let mut times = Vec::with_capacity(runs);
        let mut samples = 0;
        for run in 0..runs + 3 {
            if run == 3 && profile {
                tiny_voca::prof::enable();
            }
            let started = Instant::now();
            samples = model.synthesize(&tokens, &stress, voice, &options, None)?.audio.len();
            if run >= 3 {
                times.push(started.elapsed().as_secs_f64() * 1e3);
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        let audio = samples as f64 / model.config.sample_rate as f64;
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        if !json {
            println!("{name} tokens={:3} audio={audio:5.2}s median={median:7.2}ms min={:7.2}ms  x_realtime={:7.1}", tokens.len(), times[0], audio * 1e3 / median);
        }
        rows.push(format!("{{\"case\":\"{name}\",\"tokens\":{},\"audio_s\":{audio:.4},\"median_ms\":{median:.4},\"min_ms\":{:.4}}}", tokens.len(), times[0]));
    }
    if profile {
        tiny_voca::prof::report(runs);
    }
    if json {
        println!("{{\"engine\":\"rust-{}\",\"threads\":{threads},\"load_ms\":{load_ms:.3},\"cases\":[{}]}}", tiny_voca::ops::backend(), rows.join(","));
    } else {
        println!("load {load_ms:.1} ms, threads {threads}, backend {}", tiny_voca::ops::backend());
    }
    Ok(())
}

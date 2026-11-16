//! Phonemize lines of text.
//!
//!     cargo run --release -p dizione --example phonemize -- weights/phonemizer "Hello, it costs $3.50."
//!     cargo run --release -p dizione --example phonemize -- weights/phonemizer --jsonl < sentences.txt
//!
//! With `--jsonl` every line of standard input gives one JSON line:
//! `{"ids": [...], "stress": [...]}` or `{"error": "..."}` (the format of
//! `scripts/dizione_parity.py`). `--time` prints load and per-line timings.

use std::io::{BufRead, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(directory) = args.first() else {
        eprintln!("usage: phonemize DATA_DIR [--jsonl | --time | TEXT...]");
        std::process::exit(2);
    };
    let started = Instant::now();
    let phonemizer = dizione::Phonemizer::load(std::path::Path::new(directory)).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1);
    });
    let load = started.elapsed();
    let rest = &args[1..];
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    if rest.first().map(String::as_str) == Some("--jsonl") || rest.first().map(String::as_str) == Some("--time") {
        let timing = rest[0] == "--time";
        let lines: Vec<String> = std::io::stdin().lock().lines().map(|line| line.unwrap()).collect();
        let passes = if timing { 3 } else { 1 };
        let mut best = f64::MAX;
        let mut first = 0.0;
        for pass in 0..passes {
            let started = Instant::now();
            for line in &lines {
                // A line is a JSON string when it starts with a quote (to carry newlines and odd characters).
                let text: String = if line.starts_with('"') { serde_json_string(line) } else { line.clone() };
                let result = phonemizer.phonemize(&text);
                if !timing {
                    match result {
                        Ok(p) => writeln!(out, "{{\"ids\": {:?}, \"stress\": {:?}}}", p.ids, p.stress).unwrap(),
                        Err(error) => writeln!(out, "{{\"error\": {:?}}}", error).unwrap(),
                    }
                }
            }
            let seconds = started.elapsed().as_secs_f64();
            if pass == 0 {
                first = seconds;
            }
            best = best.min(seconds);
        }
        if timing {
            writeln!(
                out,
                "{{\"load_ms\": {:.3}, \"lines\": {}, \"first_pass_ms_per_line\": {:.4}, \"warm_ms_per_line\": {:.4}}}",
                load.as_secs_f64() * 1e3,
                lines.len(),
                first * 1e3 / lines.len().max(1) as f64,
                best * 1e3 / lines.len().max(1) as f64
            )
            .unwrap();
        }
        return;
    }
    for text in rest {
        match phonemizer.phonemize(text) {
            Ok(p) => writeln!(out, "{p}").unwrap(),
            Err(error) => writeln!(out, "error: {error}").unwrap(),
        }
    }
}

/// Minimal JSON string decoding (the escapes Python's json.dumps writes).
fn serde_json_string(line: &str) -> String {
    let line = line.trim();
    let inner = line.strip_prefix('"').unwrap_or(line);
    let inner = inner.strip_suffix('"').unwrap_or(inner);
    let mut out = String::new();
    let mut chars = inner.chars();
    let mut pending: Option<u32> = None;
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('b') => out.push('\u{8}'),
            Some('f') => out.push('\u{c}'),
            Some('u') => {
                let code = u32::from_str_radix(&chars.by_ref().take(4).collect::<String>(), 16).unwrap_or(0xFFFD);
                if (0xD800..0xDC00).contains(&code) {
                    pending = Some(code);
                } else if (0xDC00..0xE000).contains(&code) {
                    let high = pending.take().unwrap_or(0xD800);
                    out.push(char::from_u32(0x10000 + ((high - 0xD800) << 10) + (code - 0xDC00)).unwrap_or('\u{FFFD}'));
                } else {
                    out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                }
            }
            Some(other) => out.push(other),
            None => {}
        }
    }
    out
}

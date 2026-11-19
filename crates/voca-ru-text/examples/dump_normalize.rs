//! Dev aid: stdin lines of raw text -> stdout normalised text.
use std::io::{BufRead, Write};

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines().flatten() {
        writeln!(out, "{}", voca_ru_text::normalize::normalize(&line)).ok();
    }
}

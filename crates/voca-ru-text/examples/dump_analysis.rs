//! Dev aid: stdin lines of raw text -> TSV tokens / stressed / phonemes.
use std::io::{BufRead, Write};

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines().flatten() {
        match voca_ru_text::analyze(&line) {
            Ok(a) => {
                let ids = a
                    .tokens
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                writeln!(
                    out,
                    "{}\t{}\t{}\t{}",
                    ids,
                    a.stressed_text,
                    a.phonemes.join(" "),
                    a.oov as u8
                )
                .ok();
            }
            Err(e) => {
                writeln!(out, "ERROR\t{e}\t").ok();
            }
        }
    }
}

//! `cargo run -p voca-ru-text -- "текст"`

use std::io::Read;

fn main() {
    let text = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let text = if text.trim().is_empty() {
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer).ok();
        buffer
    } else {
        text
    };
    if text.trim().is_empty() {
        eprintln!("usage: voca-ru-text \"текст\"");
        std::process::exit(2);
    }
    match voca_ru_text::analyze(&text) {
        Ok(result) => {
            println!("stressed: {}", result.stressed_text);
            println!("phonemes: {}", result.phonemes.join(" "));
            println!(
                "ids: {}",
                result
                    .tokens
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            println!("oov: {}", result.oov);
        }
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}

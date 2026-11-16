//! The tokens of the Python original, frozen in `fixture.jsonl` by
//! `scripts/dizione_parity.py --fixture`, must come out of this port unchanged.
//!
//! Needs the built phonemizer data (`lexicon.tsv`, `g2p.npz`, ...): the
//! directory named by `DIZIONE_DATA`, or `weights/phonemizer` at the workspace
//! root. Without it the test is skipped with a notice.

use std::path::PathBuf;

fn data_directory() -> Option<PathBuf> {
    let directory = match std::env::var_os("DIZIONE_DATA") {
        Some(path) => PathBuf::from(path),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/phonemizer"),
    };
    directory.join("lexicon.tsv").exists().then_some(directory)
}

#[test]
fn tokens_match_the_python_original() {
    let Some(directory) = data_directory() else {
        eprintln!("skipped: no phonemizer data (set DIZIONE_DATA or download weights/phonemizer)");
        return;
    };
    let phonemizer = dizione::Phonemizer::load(&directory).expect("the phonemizer loads");
    let fixture = include_str!("fixture.jsonl");
    let (mut checked, mut failures) = (0, Vec::new());
    for line in fixture.lines().filter(|line| !line.is_empty()) {
        let row: serde_json::Value = serde_json::from_str(line).expect("a fixture line is JSON");
        let text = row["text"].as_str().expect("a fixture line has a text");
        let got = phonemizer.phonemize(text);
        checked += 1;
        if row["error"] == true {
            if got.is_ok() {
                failures.push(format!("{text:?}: Python rejects this text, Rust gave {}", got.unwrap()));
            }
            continue;
        }
        let ids: Vec<u32> = row["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let stress: Vec<u8> = row["stress"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u8).collect();
        match got {
            Ok(p) if p.ids == ids && p.stress == stress => {
                assert_eq!(p.symbols.len(), p.ids.len());
                assert!(p.symbols.iter().zip(&p.ids).all(|(symbol, id)| dizione::SYMBOLS[*id as usize] == *symbol));
            }
            Ok(p) => failures.push(format!("{text:?}: got {p}")),
            Err(error) => failures.push(format!("{text:?}: error {error}")),
        }
    }
    assert!(checked > 1000, "the fixture is too small: {checked} lines");
    assert!(failures.is_empty(), "{} of {} texts differ:\n{}", failures.len(), checked, failures[..failures.len().min(20)].join("\n"));
}

#[test]
fn empty_text_is_rejected() {
    let Some(directory) = data_directory() else { return };
    let phonemizer = dizione::Phonemizer::load(&directory).unwrap();
    assert!(phonemizer.phonemize("").is_err());
    assert!(phonemizer.phonemize("  \n").is_err());
    assert!(phonemizer.phonemize("...").is_err());
}

#[test]
fn phonemizer_is_shareable_between_threads() {
    fn check<T: Send + Sync>() {}
    check::<dizione::Phonemizer>();
}

#[test]
fn missing_data_is_an_error() {
    assert!(dizione::Phonemizer::load(std::path::Path::new("/nonexistent/dizione-data")).is_err());
}

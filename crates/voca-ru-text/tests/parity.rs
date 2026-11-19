//! F7 parity harness: the Python reference frontend vs this crate.
//!
//! `cargo test -p voca-ru-text --test parity -- --ignored --nocapture`
//!
//! Requires the F3 venv and the RUAccent dictionary (both in the shared
//! `.cache`). Override the interpreter with `VOCA_RU_PYTHON`, or point
//! `VOCA_PARITY_REFERENCE` at an existing reference TSV to run offline.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use voca_ru_text::analyze;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn find_python(root: &Path) -> Option<PathBuf> {
    if let Ok(path) = std::env::var("VOCA_RU_PYTHON") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    for base in [root.to_path_buf(), root.parent()?.parent()?.to_path_buf()] {
        let candidate = base.join(".cache/venvs/f3/bin/python");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn run_reference(root: &Path, limit: usize) -> Option<PathBuf> {
    if let Ok(path) = std::env::var("VOCA_PARITY_REFERENCE") {
        return Some(PathBuf::from(path));
    }
    let python = find_python(root)?;
    let out = std::env::temp_dir().join(format!("voca-ru-parity-{}.tsv", std::process::id()));
    let status = Command::new(&python)
        .current_dir(root)
        .env("PYTHONPATH", root.join("training"))
        .arg(root.join("tests/parity/reference.py"))
        .arg("--batch")
        .arg(root.join("data/scripts-ru/batch1.jsonl"))
        .arg("--limit")
        .arg(limit.to_string())
        .arg("--out")
        .arg(&out)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    Some(out)
}

fn read_rows(path: &Path) -> Vec<(String, String)> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|line| {
            let row: Value = serde_json::from_str(line).ok()?;
            Some((
                row["id"].as_str()?.to_string(),
                row["text"].as_str()?.to_string(),
            ))
        })
        .collect()
}

#[test]
fn fixture_matches_python() {
    let root = repo_root();
    let fixture = std::fs::read_to_string(root.join("tests/parity/fixture.jsonl")).unwrap();
    for line in fixture.lines().filter(|line| !line.trim().is_empty()) {
        let row: Value = serde_json::from_str(line).unwrap();
        let text = row["text"].as_str().unwrap();
        let expected: Vec<usize> = row["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let result = analyze(text).unwrap();
        assert_eq!(result.plain_text, row["plain"].as_str().unwrap(), "{text}");
        assert_eq!(result.tokens, expected, "{text}");
    }
}

#[test]
#[ignore = "runs the Python reference frontend; see tests/parity/README.md"]
fn parity_1000() {
    let root = repo_root();
    let batch = root.join("data/scripts-ru/batch1.jsonl");
    let rows = read_rows(&batch);
    let rows = &rows[..rows.len().min(1000)];

    let Some(reference_path) = run_reference(&root, rows.len()) else {
        panic!(
            "Python reference unavailable. Set VOCA_RU_PYTHON to the F3 venv \
             interpreter, or VOCA_PARITY_REFERENCE to a reference TSV."
        );
    };
    let reference = std::fs::read_to_string(&reference_path).unwrap();
    let mut exact = 0usize;
    let mut token_disagreements = 0usize;
    let mut total_tokens = 0usize;
    let mut stress_exact = 0usize;
    let mut python_oov = 0usize;
    let mut rust_oov = 0usize;
    let mut normalisation_causes = 0usize;
    let mut g2p_causes = 0usize;
    let mut stress_causes = 0usize;
    let mut examples: Vec<String> = Vec::new();

    let mut lines = reference.lines();
    for (id, text) in rows {
        let line = lines.next().expect("reference row");
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 6, "bad reference row: {line}");
        let expected_tokens: Vec<usize> = fields[1]
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let (expected_plain, expected_stressed, expected_phones) =
            (fields[2], fields[3], fields[4]);
        python_oov += (fields[5] == "1") as usize;

        let result = analyze(text).unwrap();
        rust_oov += result.oov as usize;
        let actual_tokens = &result.tokens;
        total_tokens += expected_tokens.len().max(actual_tokens.len());

        if *actual_tokens == expected_tokens {
            exact += 1;
            if result.stressed_text == expected_stressed {
                stress_exact += 1;
            } else {
                stress_causes += 1;
            }
            continue;
        }

        let disagreements = expected_tokens
            .iter()
            .zip(actual_tokens.iter())
            .filter(|(a, b)| a != b)
            .count()
            + expected_tokens.len().abs_diff(actual_tokens.len());
        token_disagreements += disagreements;

        let cause = if result.plain_text != expected_plain {
            normalisation_causes += 1;
            "normalisation"
        } else {
            g2p_causes += 1;
            "g2p"
        };
        if examples.len() < 3 {
            examples.push(format!(
                "{id}: {cause}, {disagreements} token(s)\n  ref: {}\n  got: {}",
                expected_phones,
                result.phonemes.join(" ")
            ));
        }
    }

    let total = rows.len();
    let exact_pct = 100.0 * exact as f64 / total as f64;
    println!("sentences: {total}");
    println!("exact token match: {exact}/{total} = {exact_pct:.2}%");
    println!(
        "token disagreements: {token_disagreements}/{total_tokens} = {:.4}%",
        100.0 * token_disagreements as f64 / total_tokens as f64
    );
    println!(
        "stress exact: {stress_exact}/{total} = {:.2}%",
        100.0 * stress_exact as f64 / total as f64
    );
    println!("oov sentences: python {python_oov}, rust {rust_oov}");
    println!(
        "causes: normalisation {normalisation_causes}, g2p {g2p_causes}, \
         stress-only {stress_causes}"
    );
    for example in &examples {
        println!("\n{example}");
    }
    assert!(
        exact_pct >= 99.0,
        "token parity {exact_pct:.2}% is below the 99% contract gate"
    );
}

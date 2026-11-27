use std::process::Command;

#[test]
fn cli_prints_stress_phonemes_and_ids() {
    let output = Command::new(env!("CARGO_BIN_EXE_voca-ru-text"))
        .arg("Привет, как дела?")
        .output()
        .expect("run voca-ru-text");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("stressed: приве́т, ка́к де́ла?"), "{stdout}");
    assert!(stdout.contains("phonemes: <bos> "), "{stdout}");
    assert!(stdout.contains("ids: 1 "), "{stdout}");
}

#[test]
fn cli_rejects_empty_input() {
    let output = Command::new(env!("CARGO_BIN_EXE_voca-ru-text"))
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run voca-ru-text");
    assert_eq!(output.status.code(), Some(2));
}

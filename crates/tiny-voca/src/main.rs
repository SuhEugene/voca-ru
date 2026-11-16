//! Command-line text to speech.
//!
//!     tiny-voca --text "Hello! This is tiny voca." --voice F2 --output hello.wav

use std::path::PathBuf;
use std::time::Instant;

use tiny_voca::tts::Tts;
use tiny_voca::Options;

const USAGE: &str = "\
Generate speech with tiny-voca.

Usage: tiny-voca --text TEXT --output FILE.wav [options]

Options:
  --text TEXT          Text to speak (any length; split into sentences)
  --text-file FILE     Read the text from a file instead ('-' for standard input)
  --output FILE        WAV file to write
  --voice NAME         F2 (default) or M2
  --speed X            Speaking rate, 0.25 to 4 (default 1)
  --pitch-shift X      Added to log F0; 0.058 is about one semitone (default 0)
  --pitch-range X      Scale pitch movement around the mean, 0 to 3 (default 1)
  --seed N             Seed of the noise source (default 0)
  --model FILE         Weights (default: $TINY_VOCA_MODEL or weights/tiny-voca.safetensors)
  --data DIR           Phonemizer data (default: $DIZIONE_DATA or weights/phonemizer)
  --threads N          Worker threads (default 1; more helps long sentences)
  --no-split           Speak the text as one utterance, like the reference CLI
  --pcm16              Write 16-bit PCM instead of 32-bit float
  --quiet              Print nothing
  -h, --help           Show this help
";

struct Arguments {
    text: Option<String>,
    text_file: Option<String>,
    output: Option<PathBuf>,
    voice: String,
    options: Options,
    model: PathBuf,
    data: PathBuf,
    threads: usize,
    pcm16: bool,
    quiet: bool,
    whole: bool,
}

fn parse() -> Result<Arguments, String> {
    let env = |name: &str, default: &str| PathBuf::from(std::env::var_os(name).unwrap_or_else(|| default.into()));
    let mut a = Arguments {
        text: None, text_file: None, output: None, voice: "F2".into(), options: Options::default(),
        model: env("TINY_VOCA_MODEL", "weights/tiny-voca.safetensors"), data: env("DIZIONE_DATA", "weights/phonemizer"),
        threads: 1, pcm16: false, quiet: false, whole: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        fn number<T: std::str::FromStr>(name: &str, text: String) -> Result<T, String> {
            text.parse().map_err(|_| format!("{name}: invalid number {text}"))
        }
        match flag.as_str() {
            "--text" => a.text = Some(value("--text")?),
            "--text-file" => a.text_file = Some(value("--text-file")?),
            "--output" | "-o" => a.output = Some(value("--output")?.into()),
            "--voice" => a.voice = value("--voice")?,
            "--speed" => a.options.speed = number("--speed", value("--speed")?)?,
            "--pitch-shift" => a.options.pitch_shift = number("--pitch-shift", value("--pitch-shift")?)?,
            "--pitch-range" => a.options.pitch_range = number("--pitch-range", value("--pitch-range")?)?,
            "--seed" => a.options.seed = number("--seed", value("--seed")?)?,
            "--model" => a.model = value("--model")?.into(),
            "--data" => a.data = value("--data")?.into(),
            "--threads" => a.threads = number::<usize>("--threads", value("--threads")?)?.max(1),
            "--pcm16" => a.pcm16 = true,
            "--no-split" => a.whole = true,
            "--quiet" | "-q" => a.quiet = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown option {other}\n\n{USAGE}")),
        }
    }
    Ok(a)
}

fn run() -> Result<(), String> {
    let started = Instant::now();
    let a = parse()?;
    let text = match (&a.text, &a.text_file) {
        (Some(text), None) => text.clone(),
        (None, Some(path)) if path == "-" => std::io::read_to_string(std::io::stdin()).map_err(|e| e.to_string())?,
        (None, Some(path)) => std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?,
        _ => return Err(format!("give exactly one of --text and --text-file\n\n{USAGE}")),
    };
    let output = a.output.ok_or_else(|| format!("--output is required\n\n{USAGE}"))?;
    // The matrix products run on our own threads, not the BLAS library's.
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    rayon::ThreadPoolBuilder::new().num_threads(a.threads).build_global().map_err(|e| e.to_string())?;
    let tts = Tts::load(&a.model, &a.data)?;
    let loaded = started.elapsed();
    let speech = if a.whole { tts.speak_whole(&text, &a.voice, &a.options)? } else { tts.speak(&text, &a.voice, &a.options)? };
    tiny_voca::wav::write(&output, &speech.audio, speech.sample_rate, a.pcm16).map_err(|e| format!("{}: {e}", output.display()))?;
    if !a.quiet {
        for utterance in &speech.utterances {
            println!("{}", utterance.symbols.join(" "));
        }
        let seconds = speech.audio.len() as f64 / speech.sample_rate as f64;
        let peak = speech.audio.iter().fold(0.0f32, |peak, v| peak.max(v.abs()));
        println!("Saved {seconds:.2}s; peak={peak:.4}");
        let model = speech.model_time.as_secs_f64();
        println!("load {:.1} ms, phonemes {:.1} ms, model {:.1} ms ({:.0}x real time), total {:.1} ms",
                 loaded.as_secs_f64() * 1e3, speech.phoneme_time.as_secs_f64() * 1e3, model * 1e3, seconds / model.max(1e-9),
                 started.elapsed().as_secs_f64() * 1e3);
    }
    Ok(())
}

fn main() {
    if let Err(message) = run() {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

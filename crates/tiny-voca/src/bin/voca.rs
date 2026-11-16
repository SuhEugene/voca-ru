//! Speak text from the terminal.
//!
//!     voca hello there, this is tiny voca
//!     echo "Read this aloud." | voca
//!     voca -o hello.wav "Save this instead of playing it."

use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::Command;

use tiny_voca::tts::Tts;
use tiny_voca::Options;

const USAGE: &str = "\
Speak text aloud with tiny-voca.

Usage: voca [options] TEXT...
       echo TEXT | voca [options]

Options:
  -v, --voice NAME     F2 (default) or M2
  -s, --speed X        Speaking rate, 0.25 to 4 (default 1)
  -p, --pitch X        Pitch shift in semitones (default 0)
  -o, --output FILE    Write a WAV file instead of playing
  -f, --file FILE      Read the text from a file
      --seed N         Seed of the noise source (default 0)
  -h, --help           Show this help

Data is read from $VOCA_HOME (default ~/.local/share/voca).
";

fn home() -> Result<PathBuf, String> {
    if let Some(dir) = std::env::var_os("VOCA_HOME") {
        return Ok(dir.into());
    }
    let home = std::env::var_os("HOME").ok_or("HOME is not set; set VOCA_HOME to the data directory")?;
    Ok(PathBuf::from(home).join(".local/share/voca"))
}

fn play(path: &std::path::Path) -> Result<(), String> {
    let players: &[&str] = if cfg!(target_os = "macos") { &["afplay"] } else { &["pw-play", "paplay", "aplay", "play"] };
    for player in players {
        match Command::new(player).arg(path).status() {
            Ok(status) if status.success() => return Ok(()),
            Ok(status) => return Err(format!("{player} failed ({status})")),
            Err(_) => continue,
        }
    }
    Err(format!("no audio player found (tried {}); use --output FILE", players.join(", ")))
}

fn run() -> Result<(), String> {
    let mut words: Vec<String> = Vec::new();
    let mut voice = String::from("F2");
    let mut options = Options::default();
    let mut output: Option<PathBuf> = None;
    let mut file: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        fn number<T: std::str::FromStr>(name: &str, text: String) -> Result<T, String> {
            text.parse().map_err(|_| format!("{name}: invalid number {text}"))
        }
        match arg.as_str() {
            "-v" | "--voice" => voice = value("--voice")?.to_uppercase(),
            "-s" | "--speed" => options.speed = number("--speed", value("--speed")?)?,
            "-p" | "--pitch" => options.pitch_shift = number::<f32>("--pitch", value("--pitch")?)? * (2f32.ln() / 12.0),
            "-o" | "--output" => output = Some(value("--output")?.into()),
            "-f" | "--file" => file = Some(value("--file")?),
            "--seed" => options.seed = number("--seed", value("--seed")?)?,
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            "--" => {
                words.extend(args.by_ref());
                break;
            }
            // Anything else is text, so `voca -5 degrees` and stray dashes still speak.
            _ => words.push(arg),
        }
    }
    let text = if let Some(path) = file {
        std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?
    } else if !words.is_empty() {
        words.join(" ")
    } else if !std::io::stdin().is_terminal() {
        std::io::read_to_string(std::io::stdin()).map_err(|e| e.to_string())?
    } else {
        print!("{USAGE}");
        return Ok(());
    };
    if text.trim().is_empty() {
        return Err("nothing to say".into());
    }

    let home = home()?;
    let model = home.join("tiny-voca.safetensors");
    if !model.exists() {
        return Err(format!("{} not found; reinstall from https://github.com/sultson/voca, or set VOCA_HOME", model.display()));
    }
    // The matrix products run on our own threads, not the BLAS library's.
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(4);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().map_err(|e| e.to_string())?;
    let tts = Tts::load(&model, &home.join("phonemizer"))?;
    let speech = tts.speak(&text, &voice, &options)?;

    if let Some(path) = output {
        return tiny_voca::wav::write(&path, &speech.audio, speech.sample_rate, true).map_err(|e| format!("{}: {e}", path.display()));
    }
    let path = std::env::temp_dir().join(format!("voca-{}.wav", std::process::id()));
    tiny_voca::wav::write(&path, &speech.audio, speech.sample_rate, true).map_err(|e| format!("{}: {e}", path.display()))?;
    let played = play(&path);
    let _ = std::fs::remove_file(&path);
    played
}

fn main() {
    if let Err(message) = run() {
        eprintln!("voca: {message}");
        std::process::exit(1);
    }
}

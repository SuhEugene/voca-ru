# voca

Type `voca hello` and your computer says it. Offline, instant, 4.5M parameters.

```sh
voca hello there, this is tiny voca
```

> **Fork of [n4ze3m/tiny-voca](https://github.com/n4ze3m/tiny-voca).** The model,
> the weights and the training are Muhammed Nazeem's work. This fork rewrites
> inference in Rust (same output, 2–7x faster) and adds the `voca` command.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/sultson/voca/main/install.sh | sh
```

That is the whole setup. No Python, no account, nothing to configure.

| System | What the installer does |
|---|---|
| macOS, Apple silicon | Downloads a ready binary (1.4 MB) and the model (19 MB) |
| Anything else | Builds with [cargo](https://rustup.rs); compiles on Linux, not yet tested there |

## Use

```sh
voca the build finished with no errors      # speak it
echo "Read this aloud." | voca              # speak piped text
voca -f notes.txt                           # speak a file
voca -v m2 -s 1.2 good morning              # male voice, 20% faster
voca -o hello.wav "Save this to a file."    # write a WAV, play nothing
```

| Option | Meaning | Default |
|---|---|---|
| `-v`, `--voice` | `f2` (female) or `m2` (male) | `f2` |
| `-s`, `--speed` | Speaking rate, 0.25 to 4 | 1 |
| `-p`, `--pitch` | Pitch shift in semitones | 0 |
| `-o`, `--output` | Write a WAV file instead of playing | play |
| `-f`, `--file` | Read the text from a file | |

English only. Quote the text if it contains `?`, `!`, `'` or `&`.

## Claude Code skill

The installer also adds a skill, so Claude can talk. Say so once:

```
> let's make this a voice session
```

From then on each reply is written in full as usual, and Claude speaks the one
or two sentences that matter most. The whole skill is
[this file](skills/voca/SKILL.md):

```
Each turn, write the full answer as usual. Then run `voca "<gist>"` in the
background: one or two plain sentences carrying only what matters most.
```

## Speed

Apple M1 Pro. Time to turn phonemes into audio, median of repeated runs.

| Audio length | PyTorch (original) | ONNX (published) | voca | voca vs PyTorch |
|---:|---:|---:|---:|---:|
| 1.9 s | 19 ms | 32 ms | **7 ms** | 2.8x |
| 6.3 s | 62 ms | 123 ms | **29 ms** | 2.1x |
| 16.8 s | 206 ms | 483 ms | **117 ms** | 1.8x |
| 16.8 s, 4 threads | 132 ms | 188 ms | **60 ms** | 2.2x |

| Whole command, text in to WAV out | Python | voca | |
|---|---:|---:|---:|
| 1.9 s of audio | 880 ms | **24 ms** | 37x |
| 6.3 s of audio | 907 ms | **45 ms** | 20x |

That is 140 to 350 times faster than real time. Method and every number:
[BENCHMARK.md](BENCHMARK.md).

## Same output as the original

- Phonemes: identical on all 1,828 test texts.
- Timing of every sound: identical on all 38 test cases.
- Waveform: differs from PyTorch only by rounding (52 to 107 dB below the signal).

## For developers

```sh
git clone https://github.com/sultson/voca && cd voca
./install.sh          # in a checkout, builds from source
```

- `crates/tiny-voca`: the model, the vocoder, the `voca` and `tiny-voca` commands.
- `crates/dizione`: Rust port of the [dizione](https://github.com/n4ze3m/dizione) phonemizer.
- `scripts/`: weight conversion, parity checks and benchmarks against upstream.

```rust
let tts = tiny_voca::tts::Tts::load("tiny-voca.safetensors".as_ref(), "phonemizer".as_ref())?;
let speech = tts.speak("Hello there.", "F2", &tiny_voca::Options::default())?;
tiny_voca::wav::write("hello.wav".as_ref(), &speech.audio, speech.sample_rate, false)?;
```

Not ported: training, evaluation, batching and the web demo. Those live
[upstream](https://github.com/n4ze3m/tiny-voca).

## Licence

Apache-2.0, as upstream. See [NOTICE](NOTICE). The pronunciation data derives
from CMUdict.

# voca-ru

Russian speech from a 4.5M-parameter model, offline, one binary.

```sh
voca "Привет, как дела?"
```

Rust inference for the Russian [voca-ru](https://huggingface.co/SuhEugene/voca-ru)
checkpoint. The network is the same tiny-voca source-filter architecture
(24 kHz, 4.5M parameters, 184 tensors) as [sultson/voca](https://github.com/sultson/voca)
and [n4ze3m/tiny-voca](https://github.com/n4ze3m/tiny-voca); the phoneme
inventory and the frontend are Russian.

## Build

Rust 1.80+ is enough; no system libraries beyond the C toolchain.

```sh
. "$(git rev-parse --show-toplevel)/scripts/cache-env.sh"   # this repo's shared caches
cargo build --release -p tiny-voca --bin voca
```

The binary is `target/release/voca`.

## Install locally

`voca` reads `$VOCA_HOME` (default `~/.local/share/voca`), which must contain
`tiny-voca.safetensors`.

```sh
mkdir -p ~/.local/share/voca
tar -xzf out/voca-data.tar.gz -C ~/.local/share/voca
cp target/release/voca ~/.local/bin/voca
```

`install.sh` expects the same archive layout at
`<release>/voca-data.tar.gz`: the safetensors file at the root and the
frontend files under `phonemizer/`.

## Use

```sh
voca "Мой коллега уходит в отпуск на две недели."
echo "Прочитай это вслух." | voca
voca -f notes.txt                          # speak a file
voca -v m2 -s 1.2 доброе утро              # male voice, 20 % faster
voca -p -2 добрый вечер                    # two semitones lower
voca -o privet.wav "Сохрани это в файл."   # write a WAV
```

Voice names: `f2` (female, default) and `m2` (male), case-insensitive.
Options: `-v/--voice`, `-s/--speed` 0.25–4, `-p/--pitch` in semitones,
`-o/--output`, `-f/--file`, `--seed`.

The recommended pitch correction is `pitch_shift ≈ -0.18` (about −3
semitones) for F2 and `≈ -0.13` (about −2.2 semitones) for M2; `-p -3` and
`-p -2` are the nearest semitone values.

The development binary `target/release/tiny-voca` has `--model`, `--data`,
`--threads`, `--no-split`, `--pitch-shift` and `--pcm16` for experiments.

## Frontend

`crates/voca-ru-text` does normalisation → RUAccent-derived stress →
eSpeak-derived G2P → the 76-symbol inventory. The stress dictionary and the
G2P lexicon are compiled into the binary, so runtime speech needs no Python,
eSpeak or network. The same files are still shipped under `phonemizer/` in the
data archive for inspection and compatibility with the installer.

Inventory layout (`training/voca_ru/text/symbols_ru.py` is authoritative):

```
0..4    5 specials     <pad> <bos> <eos> <sp> <unk>
5..27   23 punctuation
28..63  36 consonants   hard/soft pairs are separate symbols (b, bʲ, …)
64..75  12 vowels
```

`crates/tiny-voca/src/model.rs` takes the layout from the frontend crate and
checks the checkpoint's `symbols` metadata, so weights for another inventory
are rejected instead of being read with shifted ids.

## Data archive

`out/voca-data.tar.gz` (~17 MB) is built from the published checkpoint:

```sh
. "$(git rev-parse --show-toplevel)/scripts/cache-env.sh"
. .cache/venvs/r4/bin/activate            # or call its python directly
python scripts/convert_weights.py \
    .cache/models/voca-ru-run/inference.pt out/tiny-voca.safetensors
mkdir -p out/stage/phonemizer
cp out/tiny-voca.safetensors out/stage/
cp crates/voca-ru-text/src/data/stress_dict.txt \
   crates/voca-ru-text/src/data/g2p_lexicon.txt out/stage/phonemizer/
tar -czf out/voca-data.tar.gz -C out/stage .
```

`out/` is git-ignored; weights and audio are never committed.

## Quality and limits

From the model card of [SuhEugene/voca-ru](https://huggingface.co/SuhEugene/voca-ru):

- Synthetic Russian voices; intelligible, but not natural. WER 4.76 % on the
  five-phrase evaluation set.
- F2 is brighter than M2 by roughly 3 dB in 2–8 kHz.
- Both voices are pitched up by 2–3 semitones; apply the pitch correction
  above if the result sounds too high.
- 24 kHz mono, one utterance up to 512 tokens / 32 s. Longer text is split
  into sentences.
- Output is 32-bit float WAV; `voca` does not normalise the peak (it can
  slightly exceed 1.0), so normalise before 16-bit export if needed.

Parity with PyTorch on the same seed, voice and tokens: identical durations,
maximum waveform difference 6.9e-4 relative to peak, 66–83 dB SNR over ten
cases. Speed on this Mac (M-series, release build): model-only RTF 0.0029 for
a 2.4 s sentence (≈350× real time on one thread), first audio in ≈27 ms
including load and phonemisation. Details and reproduction:
[reports/rust-russian.md](reports/rust-russian.md).

## Licence

Apache-2.0; see [NOTICE](NOTICE). The Rust inference is derived from
sultson/voca and n4ze3m/tiny-voca. The Russian frontend's pronunciation data
derives from eSpeak NG (GPL-3.0) and RUAccent; see the model card and
`NOTICE`.

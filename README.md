# voca

A spoken sentence in ~30 ms. 4.5M parameters, offline, one binary.

```sh
voca hello there
```

Rust rewrite of [tiny-voca](https://github.com/n4ze3m/tiny-voca) by
[n4ze3m](https://github.com/n4ze3m), who designed and trained the model.
Inference runs 2–7x faster; a cold start, 20–37x.

## Speed

| | PyTorch (original) | ONNX | voca |
|---|---:|---:|---:|
| 2 s of audio, model only | 19 ms | 32 ms | **7 ms** |
| 6 s of audio, model only | 62 ms | 123 ms | **29 ms** |
| 17 s of audio, model only | 206 ms | 483 ms | **117 ms** |
| 2 s of audio, whole command | 880 ms | | **24 ms** |
| 6 s of audio, whole command | 907 ms | | **45 ms** |

M1 Pro, one thread, medians. Method and all numbers: [BENCHMARK.md](BENCHMARK.md).

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/sultson/voca/main/install.sh | sh
```

Apple-silicon Macs get a prebuilt binary; a 20 MB download with the model.
Other systems build with [cargo](https://rustup.rs) (Linux untested).

## Use

```sh
voca the build finished                     # speak
echo "Read this aloud." | voca              # speak piped text
voca -f notes.txt                           # speak a file
voca -v m2 -s 1.2 good morning              # male voice, 1.2x speed
voca -p -2 good evening                     # two semitones lower
voca -o hello.wav "Save this to a file."    # write a WAV
```

Voices: `f2` (default), `m2`. English only.

## Skill

[`skills/voca/SKILL.md`](skills/voca/SKILL.md) gives a coding agent a voice.
Call the session a voice session: the agent keeps writing full answers and
speaks the gist of each.

The installer copies it to `~/.claude/skills` when that folder exists.
For other agents, copy the file to wherever they load skills.

## Licence

Apache-2.0, as upstream; see [NOTICE](NOTICE). Pronunciation data derives from CMUdict.

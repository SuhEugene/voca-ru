# Voice references (D2)

`f_neutral` and `m_neutral` were chosen by the human from the G1 VoiceDesign
candidates. Each voice is a `.wav` (24 kHz, mono) plus `.txt` (the exact
transcript used as the clone prompt). The audio is git-ignored (`*.wav`); a
fresh checkout recreates it with:

```sh
.cache/venvs/r4/bin/python scripts/gen/teacher.py design
cp out/g1/voices/{f_neutral,m_neutral}.{wav,txt} assets/voices/
```

The transcripts are committed so the prompt text is reviewable. Seeds and
descriptions live in `scripts/gen/teacher.py` (`VOICES`, seeds 202 and 404).

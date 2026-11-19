"""Screen one generated clip: WER, duration/rate, level, silence, stress.

Everything here is deterministic and model-free except the two lazily built
judges (Whisper and meter2).  ``build_corpus.py`` calls ``screen_clip`` after
saving a FLAC; the metrics and the ``reject_reasons`` list go to the manifest
or to ``review.jsonl``.

    from screen import prepare_audio, screen_clip, AsrJudge, StressJudge
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAINING = os.path.join(REPO, "training")
if TRAINING not in sys.path:
    sys.path.insert(0, TRAINING)

from voca_ru.eval import wer as _wer  # noqa: E402  (F5 scoring, F3 normaliser)

CACHE = os.environ.get("VOCA_CACHE", os.path.join(REPO, ".cache"))
ACUTE = "\u0301"

# contract §2
SR = 24000
MIN_SECONDS, MAX_SECONDS = 0.5, 35.0
MIN_RATE, MAX_RATE = 0.5, 7.0          # words per second
MAX_WER = 0.15
DURATION_RATIO = (0.6, 1.6)            # vs chars / 14 chars s-1
CHARS_PER_SECOND = 14.0
PEAK_TARGET = 0.95
CLIP_PEAK = 0.999
MIN_RMS = 0.005
TRIM_THRESHOLD = 0.01                  # about -40 dBFS
TRIM_PAD = 0.05
STRESS_MARGIN = 0.5                    # meter2 confidence in its own answer
STRESS_MAX_ERROR_RATE = 0.10           # per clip, over checked multi-vowel words


# ---------------------------------------------------------------- audio shaping

def trim_silence(wav, sr=SR, threshold=TRIM_THRESHOLD, pad=TRIM_PAD):
    """Trim leading/trailing near-silence, keeping ``pad`` seconds on each side."""
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return a, 0.0, 0.0
    loud = np.flatnonzero(np.abs(a) >= threshold)
    if loud.size == 0:
        return a, 0.0, 0.0
    keep = int(round(pad * sr))
    start = max(0, int(loud[0]) - keep)
    end = min(a.size, int(loud[-1]) + 1 + keep)
    return a[start:end], start / sr, (a.size - end) / sr


def peak_normalize(wav, target=PEAK_TARGET):
    """Scale to ``target``; a silent or already-quiet clip is left alone."""
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.abs(a).max()) if a.size else 0.0
    if peak <= 0.0:
        return a, 0.0
    gain = target / peak
    return (a * gain).astype(np.float32), gain


def audio_stats(wav, sr=SR):
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.abs(a).max()) if a.size else 0.0
    rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
    return {
        "seconds": round(a.size / sr, 3),
        "peak": round(peak, 4),
        "rms": round(rms, 5),
        "clipped": bool(peak >= CLIP_PEAK),
    }


def prepare_audio(wav, sr=SR, target=PEAK_TARGET):
    """Trim, then peak-normalise. Returns (audio, stats, trim, gain)."""
    raw = audio_stats(wav, sr)
    trimmed, lead, tail = trim_silence(wav, sr)
    shaped, gain = peak_normalize(trimmed, target)
    stats = audio_stats(shaped, sr)
    stats["raw_peak"] = raw["peak"]
    stats["raw_rms"] = raw["rms"]
    stats["raw_clipped"] = raw["clipped"]
    stats["lead_trim_s"] = round(lead, 3)
    stats["tail_trim_s"] = round(tail, 3)
    stats["gain"] = round(gain, 3)
    return shaped, stats


# ---------------------------------------------------------------- stress (meter2)

_WORD_MARKED = re.compile(r"[А-Яа-яЁё" + ACUTE + r"]+(?:-[А-Яа-яЁё" + ACUTE + r"]+)*")


class StressJudge:
    """meter2, the adapter repo's audio stress detector (Apache-2.0).

    ``meter2.load_ctc`` calls ``hf_hub_download`` even for a local model path;
    the encoder is preloaded here from the shared cache instead, so no network
    is touched.  ``head.pt`` comes from the adapter download.
    """

    def __init__(self, meter_dir=None, ctc_dir=None, head=None, device="cpu"):
        meter_dir = meter_dir or os.path.join(CACHE, "models/qwen3-tts-ru-stress/meter")
        ctc_dir = ctc_dir or os.path.join(CACHE, "models/wav2vec2-xlsr-ru")
        head = head or os.path.join(meter_dir, "head.pt")
        if not os.path.isdir(ctc_dir):
            raise FileNotFoundError(f"meter2 CTC model missing: {ctc_dir}")
        if not os.path.isfile(head):
            raise FileNotFoundError(f"meter2 head missing: {head}")
        os.environ.setdefault("METER2_HEAD", head)
        os.environ.setdefault("METER2_DEVICE", device)
        if meter_dir not in sys.path:
            sys.path.insert(0, meter_dir)
        import meter2

        self._meter2 = meter2
        if meter2._Models.ctc is None:
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

            meter2._Models.ctc = Wav2Vec2ForCTC.from_pretrained(ctc_dir).eval().to(device)
            meter2._Models.proc = Wav2Vec2FeatureExtractor.from_pretrained(ctc_dir)
            meter2._Models.vocab = json.load(open(os.path.join(ctc_dir, "vocab.json")))
        self.head = head

    def judge(self, wav, sr, stressed_text):
        return self._meter2.judge(wav, sr, stressed_text, head=self.head)


def stress_check(wav, sr, stressed_text, judge):
    """Compare meter2's stressed vowel with each U+0301 mark.

    Only multi-vowel words that meter2 aligned confidently (``align`` above
    ``METER2_CONF_MIN``), scored with ``margin >= STRESS_MARGIN`` and carrying a
    mark are checked.  One-vowel words and unmarked clitics are skipped.
    """
    words = _WORD_MARKED.findall(unicodedata.normalize("NFC", stressed_text))
    verdicts = judge.judge(wav, sr, stressed_text)
    checked = errors = 0
    detail = []
    for i, (word, verdict) in enumerate(zip(words, verdicts)):
        expected = judge._meter2.marked_index(word)
        best = verdict.get("best")
        margin = verdict.get("margin")
        if expected is None or best is None or margin is None:
            continue
        vowels = judge._meter2.vowel_pos(judge._meter2.plain_text(word))
        if len(vowels) < 2 or best >= len(vowels):
            continue
        if margin < STRESS_MARGIN:
            continue
        checked += 1
        if int(best) != int(expected):
            errors += 1
            detail.append({"word": word, "expected": expected, "heard": int(best),
                           "margin": round(float(margin), 3)})
    rate = errors / checked if checked else None
    if checked <= 2:
        ok = errors == 0
    else:
        ok = rate <= STRESS_MAX_ERROR_RATE
    return {
        "stress_ok": bool(ok),
        "stress_checked": checked,
        "stress_errors": errors,
        "stress_error_rate": None if rate is None else round(rate, 3),
        "stress_detail": detail,
    }


# ---------------------------------------------------------------- ASR (Whisper)

class AsrJudge:
    def __init__(self, device="mps", compute_type="float16"):
        from voca_ru.eval import make_backend

        self.backend = make_backend(
            "faster-whisper", device=device, compute_type=compute_type, language="ru"
        )

    def transcribe(self, path):
        return self.backend.transcribe(path)


# ---------------------------------------------------------------- the screen

def screen_clip(path, *, text, stressed_text, stats, asr, stress,
                max_wer=MAX_WER):
    """Judge one saved clip. Returns metrics plus ``ok`` and ``reject_reasons``.

    ``stats`` comes from ``prepare_audio`` (duration, raw peak, …).  The audio
    at ``path`` is the already trimmed and normalised file.
    """
    reasons = []
    seconds = stats["seconds"]
    word_count = len(text.split())
    expected = max(len(text), 1) / CHARS_PER_SECOND
    ratio = seconds / expected if expected else 0.0

    if not (MIN_SECONDS <= seconds <= MAX_SECONDS):
        reasons.append(f"duration {seconds:.2f}s outside {MIN_SECONDS}-{MAX_SECONDS}s")
    if not (DURATION_RATIO[0] <= ratio <= DURATION_RATIO[1]):
        reasons.append(f"duration ratio {ratio:.2f} outside {DURATION_RATIO[0]}-{DURATION_RATIO[1]}")
    rate = word_count / seconds if seconds else 0.0
    if not (MIN_RATE <= rate <= MAX_RATE):
        reasons.append(f"speaking rate {rate:.2f} w/s outside {MIN_RATE}-{MAX_RATE}")
    if stats.get("raw_clipped"):
        reasons.append(f"clipped raw peak {stats.get('raw_peak')}")
    if stats["rms"] < MIN_RMS:
        reasons.append(f"near-silent rms {stats['rms']}")

    transcript = asr.transcribe(path)
    clip_wer = _wer(text, transcript)
    if clip_wer > max_wer:
        reasons.append(f"wer {clip_wer:.3f} > {max_wer}")

    stress_result = stress_check(
        _read_audio(path), SR, stressed_text, stress
    ) if stress is not None else {
        "stress_ok": None, "stress_checked": 0, "stress_errors": 0,
        "stress_error_rate": None, "stress_detail": [],
    }
    if stress_result["stress_ok"] is False:
        reasons.append(
            f"stress {stress_result['stress_errors']}/{stress_result['stress_checked']}"
        )

    return {
        "ok": not reasons,
        "reject_reasons": reasons,
        "document_seconds": round(seconds, 3),
        "duration_ratio": round(ratio, 3),
        "word_rate": round(rate, 2),
        "wer": round(clip_wer, 4),
        "asr_transcript": transcript,
        **{k: stats.get(k) for k in
           ("peak", "rms", "raw_peak", "raw_rms", "raw_clipped",
            "lead_trim_s", "tail_trim_s", "gain")},
        **stress_result,
    }


def _read_audio(path):
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav

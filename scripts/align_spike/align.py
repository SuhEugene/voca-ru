"""CTC forced alignment of eSpeak-ru phones against Russian audio.

Copy of the upstream `tiny_voca.align` logic (wav2vec2 phoneme recogniser,
20 ms emissions, onset offset, durations on the acoustic frame grid), minus the
pause/punctuation bookkeeping the spike does not need. Measurement only: the
script reports per-clip QC and per-phone statistics, nothing enters a model.

Run it through `calibrate.py`.
"""

from dataclasses import dataclass
import math
import time

import numpy as np

from mapping import FALLBACK, CANONICAL, EXPAND, VOICELESS, VOWELS

ALIGNER_RATE = 16000      # model input rate
ALIGNER_STRIDE = 320      # 20 ms emissions at 16 kHz
ONSET_OFFSET = -0.019     # upstream default: phone starts 19 ms before its first emission
OUT_RATE = 24000          # contract sample rate
OUT_HOP = 256             # contract hop


@dataclass(frozen=True)
class ModelInfo:
    model: object
    vocab: dict
    blank: int
    device: str

    def label(self, phone):
        """eSpeak phone -> (label string, how it was decided)."""
        if phone in CANONICAL:
            return CANONICAL[phone], "canonical"
        if phone in self.vocab:
            return phone, "exact"
        if phone in FALLBACK:
            return FALLBACK[phone], "fallback"
        raise KeyError(f"no aligner label for {phone!r}")

    def label_id(self, phone):
        label, kind = self.label(phone)
        return self.vocab[label], kind


def load(model_dir, device="cpu"):
    import json
    from pathlib import Path
    import torch
    from transformers import Wav2Vec2ForCTC

    torch_device = torch.device(device)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir, local_files_only=True).to(torch_device).eval()
    vocab = json.loads((Path(model_dir) / "vocab.json").read_text(encoding="utf-8"))
    return ModelInfo(model, vocab, int(model.config.pad_token_id), str(torch_device))


def log_probs(info, wave16, normalize=True):
    import torch

    audio = np.asarray(wave16, dtype=np.float64)
    if normalize:  # upstream: zero mean, unit variance
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
    with torch.inference_mode():
        values = torch.from_numpy(audio.astype(np.float32))[None].to(info.device, info.model.dtype)
        return info.model(values).logits[0].float().log_softmax(-1).cpu().numpy()


def ctc_viterbi(log_probs_matrix, labels, blank=0):
    """Best CTC path; returns per-frame extended-state indices. Numpy reference."""
    frames, count = log_probs_matrix.shape[0], len(labels)
    states = 2 * count + 1
    extended = np.full(states, blank, dtype=np.int64)
    extended[1::2] = labels
    skip = np.zeros(states, dtype=bool)
    skip[3::2] = np.asarray(labels)[1:] != np.asarray(labels)[:-1]
    score = np.full(states, -np.inf)
    score[0] = log_probs_matrix[0, blank]
    score[1] = log_probs_matrix[0, extended[1]]
    back = np.zeros((frames, states), dtype=np.int8)
    for t in range(1, frames):
        one = np.concatenate(([-np.inf], score[:-1]))
        two = np.concatenate(([-np.inf, -np.inf], score[:-2]))
        two[~skip] = -np.inf
        options = np.stack((score, one, two))
        choice = options.argmax(0)
        score = options[choice, np.arange(states)] + log_probs_matrix[t, extended]
        back[t] = choice
    state = states - 1 if score[states - 1] >= score[states - 2] else states - 2
    if not np.isfinite(score[state]):
        raise ValueError("audio is too short for this phoneme sequence")
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= back[t, state]
    return path


def token_spans(log_probs_matrix, labels, blank=0):
    """Inclusive (first, last) emission frame of each label on the best path."""
    try:
        import torch
        from torchaudio.functional import forced_align
        path, _ = forced_align(
            torch.from_numpy(np.ascontiguousarray(log_probs_matrix, dtype=np.float32))[None],
            torch.tensor([list(labels)], dtype=torch.long), blank=blank)
        path = path[0].tolist()
    except ImportError:
        path = ctc_viterbi(log_probs_matrix, labels, blank).tolist()
    spans, previous = [], blank
    for frame, label in enumerate(path):
        if label != blank and (previous == blank or label != previous):
            spans.append([frame, frame])
        elif label != blank:
            spans[-1][1] = frame
        previous = label
    if len(spans) != len(labels):
        raise ValueError(f"forced alignment visited {len(spans)} of {len(labels)} phones")
    return [(a, b) for a, b in spans]


def resample(wave, sr, target=ALIGNER_RATE):
    if sr == target:
        return np.asarray(wave, dtype=np.float64)
    from scipy.signal import resample_poly
    factor = math.gcd(sr, target)
    return resample_poly(np.asarray(wave, dtype=np.float64), target // factor, sr // factor)


def frame_voicing(wave, sr=OUT_RATE, hop=OUT_HOP, window=1024, fmin=60, fmax=400):
    """Cheap autocorrelation voicing per hop frame. Rough proxy for pyin: only
    shares are reported, and both `vowel_voiced` and the stressed-vowel check
    use the same detector, so thresholds are self-consistent."""
    frames = len(wave) // hop
    half = window // 2
    x = np.pad(np.asarray(wave, dtype=np.float64), (half, half + window))
    starts = np.arange(frames) * hop
    win = np.lib.stride_tricks.sliding_window_view(x, window)[starts]
    win = win - win.mean(axis=1, keepdims=True)
    energy = (win ** 2).sum(axis=1) + 1e-12
    lag_min, lag_max = sr // fmax, sr // fmin
    spectrum = np.fft.rfft(win, n=2 * window, axis=1)
    corr = np.fft.irfft(spectrum * np.conjugate(spectrum), axis=1)[:, :window]
    corr = corr / energy[:, None]
    peak = corr[:, lag_min:lag_max].max(axis=1)
    return np.clip(peak, 0, 1), energy


def align_phones(info, words, wave, sr, onset_offset=ONSET_OFFSET):
    """Align one utterance. `words` is eSpeak output: list of [(phone, stress)].

    Returns a result dict; raises ValueError when the transcript cannot be
    aligned (unknown phone, too long/short).
    """
    phones = [p for word in words for p, _ in word]
    stress = [s for word in words for _, s in word]
    if not phones:
        raise ValueError("empty phoneme sequence")
    label_ids, kinds, parts = [], [], []
    for phone in phones:
        if phone in EXPAND:
            if any(part not in info.vocab for part in EXPAND[phone]):
                raise ValueError(f"expanded phone {phone!r} has a label outside vocab.json")
            parts.append([info.vocab[part] for part in EXPAND[phone]])
            kinds.append("expand")
        else:
            label, kind = info.label(phone)
            if label not in info.vocab:
                raise ValueError(f"label {label!r} for {phone!r} is not in vocab.json")
            parts.append([info.vocab[label]])
            kinds.append(kind)
    label_ids = [label for group in parts for label in group]

    start = time.perf_counter()
    wave16 = resample(wave, sr, ALIGNER_RATE)
    logp = log_probs(info, wave16)
    flat_spans = token_spans(logp, label_ids, info.blank)
    spans, cursor = [], 0
    for group in parts:
        spans.append((flat_spans[cursor][0], flat_spans[cursor + len(group) - 1][1]))
        cursor += len(group)
    inference = time.perf_counter() - start

    wave24 = resample(wave, sr, OUT_RATE)
    frames = len(wave24) // OUT_HOP
    if frames < 1:
        raise ValueError("audio shorter than one frame")
    total = frames * OUT_HOP / OUT_RATE
    onsets = np.array([first * ALIGNER_STRIDE / ALIGNER_RATE + onset_offset for first, _ in spans])
    onsets = np.clip(onsets, 0.0, total)
    bounds = np.append(onsets, total)
    # durations on the 24 kHz grid, every phone at least one frame, sum exact
    edges = np.round(bounds * OUT_RATE / OUT_HOP).astype(np.int64)
    edges = np.clip(edges, 0, frames)
    edges[0] = 0
    for k in range(1, len(edges)):
        edges[k] = max(edges[k], edges[k - 1] + 1)
    if edges[-1] > frames:
        raise ValueError("audio is too short for this phoneme sequence")
    edges[-1] = frames
    durations = np.diff(edges)

    prob = np.array([float(np.mean([np.exp(logp[a:b + 1, lbl].mean()) for lbl in group]))
                     for (a, b), group in zip(spans, parts)])
    voiced, energy = frame_voicing(wave24)
    vowel_like = np.array([p in VOWELS for p in phones])
    voiceless_like = np.array([p in VOICELESS for p in phones])
    vowel_slices = [voiced[edges[i] + 1:edges[i + 1] - 1] for i in range(len(phones))]
    vv = [float(s.mean()) for s, is_v in zip(vowel_slices, vowel_like) if is_v and s.size]
    vc = [float(s.mean()) for s, is_c in zip(vowel_slices, voiceless_like) if is_c and s.size]
    stressed = [i for i, s in enumerate(stress) if s == 1]
    stressed_short = [i for i in stressed if durations[i] < 2]
    stressed_unvoiced = [i for i in stressed if vowel_like[i] and vowel_slices[i].size
                         and float(vowel_slices[i].mean()) < 0.45]

    # calibration: per-phone mean log-prob for every candidate label + top-5 model labels
    calibration = []
    for (a, b), phone, kind in zip(spans, phones, kinds):
        span_lp = logp[a:b + 1].mean(axis=0)
        candidates = {}
        names = (phone, CANONICAL.get(phone), FALLBACK.get(phone), *EXPAND.get(phone, ()))
        for name in names:
            if name and name in info.vocab:
                candidates[name] = float(span_lp[info.vocab[name]])
        top = np.argsort(span_lp)[::-1][:5]
        inv = {v: k for k, v in info.vocab.items()}
        calibration.append({
            "phone": phone,
            "kind": kind,
            "candidates": candidates,
            "top": [(inv[int(i)], round(float(span_lp[i]), 3)) for i in top],
            "frames": int(b - a + 1),
        })

    speech_frames = int((edges[-1] - edges[0]))
    return {
        "phones": phones,
        "stress": stress,
        "durations_frames": durations.tolist(),
        "durations_ms": (durations * OUT_HOP / OUT_RATE * 1000).round(1).tolist(),
        "seconds": round(total, 3),
        "probability": prob.round(4).tolist(),
        "qc": {
            "token_probability_mean": round(float(prob.mean()), 4),
            "weak_phone_fraction": round(float((prob < 0.1).mean()), 4),
            "one_frame_phone_fraction": round(float((durations == 1).mean()), 4),
            "max_phone_frames": int(durations.max()),
            "phones_per_second": round(len(phones) / max(speech_frames * OUT_HOP / OUT_RATE, 1e-6), 2),
            "token_coverage": round(float(speech_frames / frames), 4),
            "vowel_voiced": round(float(np.mean(vv)), 4) if vv else None,
            "voiceless_voiced": round(float(np.mean(vc)), 4) if vc else None,
            "stressed_vowels": len(stressed),
            "stressed_short": len(stressed_short),
            "stressed_unvoiced": len(stressed_unvoiced),
            "kinds": {k: kinds.count(k) for k in set(kinds)},
        },
        "calibration": calibration,
        "inference_seconds": round(inference, 3),
        "total_seconds": round(time.perf_counter() - start, 3),
    }

"""WER/CER without jiwer: token/char Levenshtein over normalised text."""

from __future__ import annotations

from .scoring import score_text


def _distance(ref: list, hyp: list) -> tuple[int, int, int]:
    """Levenshtein distance with substitution/insertion/deletion counts."""
    # rows: (cost, subs, dels, ins); tuple compares cost first.
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            sub = prev[j - 1][0] + (ref[i - 1] != hyp[j - 1])
            dele = prev[j][0] + 1
            ins = cur[j - 1][0] + 1
            best = min(sub, dele, ins)
            if best == sub:
                _, s, d, n = prev[j - 1]
                cur.append((best, s + (ref[i - 1] != hyp[j - 1]), d, n))
            elif best == dele:
                _, s, d, n = prev[j]
                cur.append((best, s, d + 1, n))
            else:
                _, s, d, n = cur[j - 1]
                cur.append((best, s, d, n + 1))
        prev = cur
    return prev[-1][1], prev[-1][2], prev[-1][3]


def edit_counts(ref_text: str, hyp_text: str) -> tuple[int, int, int, int]:
    """(errors, ref_len, substitutions, insertions) on normalised words."""
    ref = score_text(ref_text).split()
    hyp = score_text(hyp_text).split()
    subs, dels, ins = _distance(ref, hyp)
    return subs + dels + ins, len(ref), subs, ins


def wer(ref_text: str, hyp_text: str) -> float:
    errors, ref_len, _, _ = edit_counts(ref_text, hyp_text)
    return errors / ref_len if ref_len else float(bool(errors))


def cer(ref_text: str, hyp_text: str) -> float:
    ref = score_text(ref_text).replace(" ", "")
    hyp = score_text(hyp_text).replace(" ", "")
    if not ref:
        return float(bool(hyp))
    subs, dels, ins = _distance(list(ref), list(hyp))
    return (subs + dels + ins) / len(ref)

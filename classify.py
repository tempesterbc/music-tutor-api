"""
classify.py - classify a student's problems against the professional corridor.

Reliable categories (score-free, corridor-based):
  tempo       - too fast / too slow overall (from the DTW alignment slope)
  intonation  - flat / sharp (sustained)
  dynamics    - too loud / too soft vs pros
  tone        - brighter/harsher or duller/covered (spectral centroid)
  timbre      - breathier/airier or more pressed (spectral flatness)

Experimental (reported with a caveat; needs real-audio calibration):
  vibrato     - much wider/narrower than pros, or absent where pros use it

NOT reported here: local rushing/dragging inside the piece. Chroma-DTW does
locate the student on the professional timeline, but not precisely enough to
localise timing. Measured on synthetic takes with planted 0.4-0.5 s timing
faults, the peak local-tempo deviation of clean playing (median 12%, max 33%)
overlaps the faulted takes (median 23%, min 13%) almost completely: no threshold
gives useful sensitivity without flagging clean playing. Per-note rhythm is
handled on the front end instead, where an uploaded score supplies exact
expected onsets. `local_tempo_ratio` is kept in analysis_core for that work.
"""
import numpy as np
from collections import defaultdict

from analysis_core import (warp_map, warp_curve, warp_times, overall_tempo_ratio)
from descriptors import vibrato_profile

K = 2.0
MIN_RUN = 12          # ~0.4 s: only sustained problems, not boundary blips
MERGE_GAP = 3
MIN_DUR_S = 0.20

# Detection floors: a deviation smaller than this is never reported even if the
# professionals happen to agree very tightly at that instant. These are
# perceptual "don't bother" levels, not statistics - keep them visible and
# tunable rather than buried, because for pitch and brightness they, not the
# corridor width, are what usually decides whether something is flagged.
FLOORS = {"cents": 20.0, "db": 4.0, "centroid": 2.4, "flatness": 4.5}


def _rmed(x, k=9):
    from analysis_core import _roll_median
    return _roll_median(x, k)


def _runs(mask, min_len=MIN_RUN, gap=MERGE_GAP):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    groups, a, p = [], idx[0], idx[0]
    for k in idx[1:]:
        if k - p <= gap:
            p = k
        else:
            groups.append((a, p)); a = p = k
    groups.append((a, p))
    return [(x, y) for x, y in groups if y - x + 1 >= min_len]


def _level(dev, a, b):
    s = dev[a:b + 1]; s = s[~np.isnan(s)]
    return float(np.median(s)) if len(s) else 0.0


def align(student, corridor):
    """Align the student onto the corridor timeline once, and hand back
    everything downstream needs. The plot used to recompute this DTW from
    scratch, which doubled the alignment cost of every request."""
    m = warp_map(corridor["ref_chroma"], student["chroma"])
    n = corridor["n"]
    return {
        "map": m,
        "curves": {k: warp_curve(student[k], m, n)
                   for k in ("cents", "db", "centroid", "flatness")},
        "times": warp_times(student["times"], m, n),
    }


def classify(student, corridor, pros, alignment=None):
    n = corridor["n"]
    al = alignment or align(student, corridor)
    m, warped = al["map"], al["curves"]
    valid = corridor["valid"]

    def stu_span(a, b):
        idxs = []
        for rf in range(a, b + 1):
            idxs += m.get(rf, [])
        if not idxs:
            return np.nan, np.nan
        ts = student["times"][idxs]
        return float(ts.min()), float(ts.max())

    findings = []
    specs = [
        ("intonation", "cents", "cents_mean", "cents_std", ("sharp", "flat"), "cents"),
        ("dynamics", "db", "db_mean", "db_std", ("louder", "softer"), "dB"),
        ("tone", "centroid", "centroid_mean", "centroid_std",
         ("brighter/harsher", "duller/covered"), "st"),
        ("timbre", "flatness", "flatness_mean", "flatness_std",
         ("breathier/airier", "more pressed/edgy"), "dB"),
    ]
    for cat, key, mk, sk, (hi, lo), unit in specs:
        floor = FLOORS[key]
        dev = _rmed(warped[key] - corridor[mk])
        thr = np.maximum(K * corridor[sk], floor)
        mask = (np.abs(dev) > thr) & valid & ~np.isnan(dev)
        for a, b in _runs(mask):
            t0, t1 = stu_span(a, b)
            dur_ref = (b - a) / corridor["frame_rate"]
            if np.isnan(t0) or dur_ref < MIN_DUR_S:
                continue
            d = _level(dev, a, b)
            # severity = how far outside the corridor, in units of the threshold
            # that actually applied here (not the fixed floor, which is often
            # not the binding constraint)
            local_thr = np.nanmedian(thr[a:b + 1])
            if not np.isfinite(local_thr) or local_thr <= 0:
                local_thr = floor
            findings.append({"category": cat, "label": hi if d > 0 else lo,
                             "t0": t0, "t1": t1,
                             "pos_pct": (100.0 * a / n, 100.0 * b / n),
                             "value": "%+.0f %s" % (d, unit),
                             "severity": abs(d) / local_thr})

    # Overall tempo from the alignment slope, so it survives clip truncation and
    # ignores leading/trailing silence.
    tr = overall_tempo_ratio(corridor["times"], al["times"], valid)
    if not np.isfinite(tr) or tr <= 0:
        tr = 1.0

    # keep only the most significant findings per category/direction so the
    # report stays concise (avoids six near-identical "brighter tone" rows)
    _by = defaultdict(list)
    for f in findings:
        _by[(f["category"], f["label"])].append(f)
    kept = []
    for group in _by.values():
        group.sort(key=lambda f: abs(f.get("severity", 1.0)), reverse=True)
        kept.extend(group[:2])
    findings = kept
    findings.sort(key=lambda f: f["pos_pct"][0])

    summary = {"tempo_ratio": float(tr),
               "truncated": bool(student.get("truncated") or
                                 any(p.get("truncated") for p in pros)),
               "n_pros": len(pros)}

    # vibrato: raw held notes, student vs pro baseline (EXPERIMENTAL)
    fr = corridor["frame_rate"]
    sv = vibrato_profile(student["cents"], fr)
    pe = [vibrato_profile(p["cents"], fr)["extent"] for p in pros]
    pe = [x for x in pe if not np.isnan(x)]
    vib = []
    if pe:
        pmed = float(np.median(pe))
        if not np.isnan(sv["extent"]):
            if sv["extent"] > 1.9 * pmed:
                vib.append("vibrato much wider than professionals (~%.0f vs ~%.0f cents)"
                           % (sv["extent"], pmed))
            elif sv["n_vib"] == 0 or sv["extent"] < 0.45 * pmed:
                vib.append("little/no vibrato where professionals use it")
    summary["vibrato_msgs"] = vib
    summary["vibrato_detail"] = sv
    return findings, summary


def format_report(findings, summary):
    L = []
    tr = summary["tempo_ratio"]
    if abs(tr - 1) > 0.05:
        L.append("TEMPO: about %d%% %s than the professional average."
                 % (round(abs(tr - 1) * 100), "slower" if tr > 1 else "faster"))
    else:
        L.append("TEMPO: within the professional range.")
    for msg in summary["vibrato_msgs"]:
        L.append("VIBRATO (experimental): %s." % msg)
    if not findings and not summary["vibrato_msgs"]:
        L.append("No sustained tone/intonation/dynamics problems - inside the corridor.")
    for f in findings:
        L.append("- [%s] %s (%s) at %.1f-%.1f s (~%.0f-%.0f%% through)"
                 % (f["category"].upper(), f["label"], f["value"], f["t0"], f["t1"],
                    f["pos_pct"][0], f["pos_pct"][1]))
    return "\n".join(L)

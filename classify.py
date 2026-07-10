"""
classify.py - classify a student's problems against the professional corridor.

Reliable categories (score-free, corridor-based):
  tempo       - too fast / too slow overall
  intonation  - flat / sharp (sustained)
  dynamics    - too loud / too soft vs pros
  tone        - brighter/harsher or duller/covered (spectral centroid)
  timbre      - breathier/airier or more pressed (spectral flatness)

Experimental (reported with a caveat; needs real-audio calibration):
  vibrato     - much wider/narrower than pros, or absent where pros use it

Precise per-note RHYTHM (rushing/dragging/uneven) is handled more accurately by
the score-based module pipeline.py, which uses onset detection against a known
score; DTW timing on a score-free corridor is too noisy to localise reliably.
"""
import numpy as np
from analysis_core import warp_map, warp_curve
from descriptors import vibrato_profile

K = 2.0
MIN_RUN = 12          # ~0.28 s: only sustained problems, not boundary blips
MERGE_GAP = 3
MIN_DUR_S = 0.20


def _rmed(x, k=9):
    out = np.full_like(np.asarray(x, float), np.nan)
    h = k // 2
    for i in range(len(x)):
        s = x[max(0, i - h): i + h + 1]
        s = s[~np.isnan(s)]
        if len(s):
            out[i] = np.median(s)
    return out


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


def classify(student, corridor, pros):
    n = corridor["n"]
    m = warp_map(corridor["ref_chroma"], student["chroma"])
    warped = {k: warp_curve(student[k], m, n)
              for k in ("cents", "db", "centroid", "flatness")}
    valid = corridor["valid"]

    def stu_span(a, b):
        idxs = []
        for rf in range(a, b + 1):
            idxs += m.get(rf, [])
        if not idxs:
            return np.nan, np.nan
        ts = [student["times"][i] for i in idxs]
        return float(min(ts)), float(max(ts))

    findings = []
    specs = [
        ("intonation", "cents", "cents_mean", "cents_std", 20.0, ("sharp", "flat"), "cents"),
        ("dynamics", "db", "db_mean", "db_std", 4.0, ("louder", "softer"), "dB"),
        ("tone", "centroid", "centroid_mean", "centroid_std", 1.3,
         ("brighter/harsher", "duller/covered"), "st"),
        ("timbre", "flatness", "flatness_mean", "flatness_std", 3.0,
         ("breathier/airier", "more pressed/edgy"), "dB"),
    ]
    for cat, key, mk, sk, floor, (hi, lo), unit in specs:
        dev = _rmed(warped[key] - corridor[mk])
        thr = np.maximum(K * corridor[sk], floor)
        mask = (np.abs(dev) > thr) & valid & ~np.isnan(dev)
        for a, b in _runs(mask):
            t0, t1 = stu_span(a, b)
            dur_ref = (b - a) / corridor["frame_rate"]
            if np.isnan(t0) or dur_ref < MIN_DUR_S:
                continue
            d = _level(dev, a, b)
            findings.append({"category": cat, "label": hi if d > 0 else lo,
                             "t0": t0, "t1": t1,
                             "pos_pct": (100.0 * a / n, 100.0 * b / n),
                             "value": "%+.0f %s" % (d, unit),
                             "severity": abs(d) / (floor if floor else 1)})
    findings.sort(key=lambda f: f["pos_pct"][0])

    summary = {"tempo_ratio": student["times"][-1] / corridor["times"][-1]}

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

"""
accuracy_battery.py - quantitative accuracy validation for the analysis engine.

Runs the classifier (classify.py) against synthetic student recordings with
KNOWN, controlled faults (perf_synth.py) and measures how well it recovers
them: detection recall as a function of true fault size, false-positive rate
on clean playing, localisation tightness, and tempo-ratio error.

Why synthetic and not real recordings: real playing has no ground truth for
"how many cents flat was that note" or "how much softer", so it cannot
produce a recall/precision curve - only a qualitative sanity check (see
README in this folder). Synthetic ground truth is the standard way to
validate a signal-processing detector's sensitivity in MIR research, and is
what perf_synth.py exists for (see its own docstring: "no copyrighted
audio").

Run:  python tests/accuracy_battery.py [--out tests/accuracy_report.json]

This is deliberately NOT pytest - it is a measurement tool that takes a
couple of minutes and produces numbers for the writeup. `tests/test_accuracy.py`
holds fast pytest regression checks that reuse these helpers.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_synth import render, PHRASE, SR as SYNTH_SR          # noqa: E402
from analysis_core import extract, build_corridor               # noqa: E402
import classify as CL                                           # noqa: E402
from descriptors import vibrato_profile                         # noqa: E402

N = len(PHRASE)
DURS = [b for _, b in PHRASE]

# direction labels classify.py assigns for a positive vs negative deviation,
# keyed the same way as classify.py's `specs` table.
LABELS = {
    "intonation": ("sharp", "flat"),
    "dynamics": ("louder", "softer"),
    "tone": ("brighter/harsher", "duller/covered"),
    "timbre": ("breathier/airier", "more pressed/edgy"),
}
KEY_OF = {"intonation": "cents", "dynamics": "db", "tone": "centroid", "timbre": "flatness"}


# --------------------------------------------------------------------------
# synthesis helpers
# --------------------------------------------------------------------------
def clean_arrays():
    """The per-note control arrays for an unfaulted, default-shaped phrase.
    Always passed explicitly (never None) so every trial consumes perf_synth's
    RNG in the same sequence - only the one faulted note differs between a
    clean trial and a faulted one."""
    arch = np.sin(np.linspace(0.15, np.pi - 0.15, N))
    return dict(
        dyn=0.5 + 0.5 * arch,
        detune=np.zeros(N),
        bright=np.zeros(N),
        breath=np.zeros(N),
        vext=np.full(N, 14.0),
        vjit=np.full(N, 0.06),
        toff=np.zeros(N),
    )


def nominal_note_times(bpm):
    """Ideal (no-jitter) start/end time of each note at the given tempo -
    matches perf_synth's own timing formula minus the small random jitter."""
    beat = 60.0 / bpm
    t, out = 0.0, []
    for b in DURS:
        start = t
        dur = b * beat * 0.92
        out.append((start, start + dur))
        t += b * beat
    return out


def render_extract(bpm, seed, **overrides):
    arrs = clean_arrays()
    arrs.update(overrides)
    y = render(bpm=bpm, seed=seed, **arrs)
    return extract(y, sr=SYNTH_SR)


def faulted(base_idx, **fault):
    """clean_arrays() with one control array's value at base_idx overridden."""
    arrs = clean_arrays()
    for k, v in fault.items():
        arrs[k][base_idx] = v
    return arrs


# --------------------------------------------------------------------------
# corridor
# --------------------------------------------------------------------------
def build_pros(n=5, base_bpm=66.0, jitter=3.0, seed0=100):
    rng = np.random.default_rng(seed0)
    bpms = base_bpm + rng.uniform(-jitter, jitter, n)
    pros = [render_extract(bpm=float(bpm), seed=seed0 + i) for i, bpm in enumerate(bpms)]
    return pros, [float(b) for b in bpms]


def ref_frame_range(corridor, ref_bpm, note_idx):
    t0, t1 = nominal_note_times(ref_bpm)[note_idx]
    times = corridor["times"]
    idx = np.where((times >= t0) & (times <= t1))[0]
    if len(idx) == 0:
        idx = np.where((times >= t0 - 0.05) & (times <= t1 + 0.05))[0]
    if len(idx) == 0:
        mid = 0.5 * (t0 + t1)
        idx = np.array([int(np.argmin(np.abs(times - mid)))])
    return int(idx.min()), int(idx.max())


def measured_deviation(corridor, alignment, key, a, b):
    dev = CL._rmed(alignment["curves"][key] - corridor[key + "_mean"])
    seg = dev[a:b + 1]
    seg = seg[~np.isnan(seg)]
    return float(np.median(seg)) if len(seg) else float("nan")


def _iou(p, q):
    inter = max(0.0, min(p[1], q[1]) - max(p[0], q[0]))
    union = max(p[1], q[1]) - min(p[0], q[0])
    return inter / union if union > 0 else 0.0


def check_detected(findings, cat, a, b, n, true_dev, iou_thr=0.15):
    if not np.isfinite(true_dev) or true_dev == 0:
        return None  # no ground-truth deviation to check against
    true_pct = (100.0 * a / n, 100.0 * b / n)
    want_label = LABELS[cat][0] if true_dev > 0 else LABELS[cat][1]
    best = None
    for f in findings:
        if f["category"] != cat or f["label"] != want_label:
            continue
        iou = _iou(f["pos_pct"], true_pct)
        if iou > iou_thr and (best is None or iou > best[1]):
            best = (f, iou)
    return best  # None if missed, else (finding, iou)


# --------------------------------------------------------------------------
# per-category sweeps
# --------------------------------------------------------------------------
def sweep_corridor_category(cat, note_idx, param, values, seeds, corridor, pros, ref_bpm):
    """Generic sweep for the 4 corridor-based categories (intonation, dynamics,
    tone, timbre). `param` is the perf_synth control array name; `values` the
    raw values to set at note_idx (one perturbed note per trial)."""
    key = KEY_OF[cat]
    a, b = ref_frame_range(corridor, ref_bpm, note_idx)
    rows = []
    for v in values:
        for s in seeds:
            arrs = faulted(note_idx, **{param: v})
            student = render_extract(bpm=ref_bpm, seed=s, **arrs)
            al = CL.align(student, corridor)
            findings, _ = CL.classify(student, corridor, pros, alignment=al)
            dev = measured_deviation(corridor, al, key, a, b)
            hit = check_detected(findings, cat, a, b, corridor["n"], dev)
            rows.append({"param_value": v, "seed": s, "true_dev": dev,
                         "detected": hit is not None,
                         "iou": hit[1] if hit else None})
    return rows


def sweep_tempo(bpms, seeds, corridor, pros, ref_bpm_mean):
    rows = []
    for bpm in bpms:
        for s in seeds:
            student = render_extract(bpm=bpm, seed=s)
            al = CL.align(student, corridor)
            _, summary = CL.classify(student, corridor, pros, alignment=al)
            expected = ref_bpm_mean / bpm
            measured = summary["tempo_ratio"]
            rows.append({"bpm": bpm, "seed": s, "expected_ratio": expected,
                         "measured_ratio": measured,
                         "rel_error_pct": 100.0 * abs(measured - expected) / expected})
    return rows


def sweep_vibrato(factors, seeds, corridor, pros, ref_bpm, base_extent=14.0):
    rows = []
    for f in factors:
        for s in seeds:
            arrs = clean_arrays()
            arrs["vext"] = arrs["vext"] * f
            student = render_extract(bpm=ref_bpm, seed=s, **arrs)
            al = CL.align(student, corridor)
            _, summary = CL.classify(student, corridor, pros, alignment=al)
            true_extent = base_extent * f
            measured_extent = summary["vibrato_detail"]["extent"]
            msgs = summary["vibrato_msgs"]
            flagged_wide = any("wider" in m for m in msgs)
            flagged_narrow = any("little/no" in m for m in msgs)
            rows.append({"factor": f, "seed": s, "true_extent_cents": true_extent,
                         "measured_extent_cents": measured_extent,
                         "flagged_wide": flagged_wide, "flagged_narrow": flagged_narrow})
    return rows


def sweep_fp_by_n_pros(n_pros_list, n_clean, base_bpm=66.0, jitter=3.0,
                        pro_seed0=100, clean_seed0=700):
    """How much does the false-positive rate on clean playing improve with
    more professional references? Builds a fresh corridor for each pro count
    (drawing from the same seed stream so larger sets are supersets of
    smaller ones) and re-runs the SAME clean student trials against each,
    holding the students fixed so only the corridor changes."""
    rng = np.random.default_rng(clean_seed0)
    bpms_c = base_bpm + rng.uniform(-jitter, jitter, n_clean)
    students = [render_extract(bpm=float(b), seed=clean_seed0 + i)
                for i, b in enumerate(bpms_c)]

    out = {}
    max_n = max(n_pros_list)
    all_pros, all_bpms = build_pros(n=max_n, base_bpm=base_bpm, jitter=jitter, seed0=pro_seed0)
    for k in n_pros_list:
        pros_k = all_pros[:k]
        corridor_k = build_corridor(pros_k)
        any_fp = 0
        for student in students:
            al = CL.align(student, corridor_k)
            findings, _ = CL.classify(student, corridor_k, pros_k, alignment=al)
            if findings:
                any_fp += 1
        out[k] = any_fp / n_clean
    return out


def sweep_clean(n_trials, corridor, pros, base_bpm, jitter, seed0):
    rng = np.random.default_rng(seed0)
    bpms = base_bpm + rng.uniform(-jitter, jitter, n_trials)
    rows = []
    for i, bpm in enumerate(bpms):
        student = render_extract(bpm=float(bpm), seed=seed0 + i)
        al = CL.align(student, corridor)
        findings, summary = CL.classify(student, corridor, pros, alignment=al)
        by_cat = {}
        for f in findings:
            by_cat.setdefault(f["category"], 0)
            by_cat[f["category"]] += 1
        rows.append({"bpm": float(bpm), "seed": seed0 + i,
                     "n_findings": len(findings), "by_category": by_cat,
                     "vibrato_msgs": summary["vibrato_msgs"],
                     "tempo_ratio": summary["tempo_ratio"]})
    return rows


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def summarize_sweep(rows):
    """Group by param_value -> recall + mean |true_dev| + mean IoU of hits."""
    by_val = {}
    for r in rows:
        by_val.setdefault(r["param_value"], []).append(r)
    out = []
    for v, rs in sorted(by_val.items()):
        devs = [r["true_dev"] for r in rs if np.isfinite(r["true_dev"])]
        hits = [r for r in rs if r["detected"]]
        ious = [r["iou"] for r in hits if r["iou"] is not None]
        out.append({
            "param_value": v,
            "mean_abs_true_dev": float(np.mean(np.abs(devs))) if devs else None,
            "n": len(rs),
            "recall": len(hits) / len(rs) if rs else None,
            "mean_iou_on_hit": float(np.mean(ious)) if ious else None,
        })
    return out


def threshold_crossing(summary_rows, target=0.5):
    """Smallest mean |true_dev| at which recall first reaches `target`.

    Sorts by the MEASURED deviation, not by the raw sweep parameter - for
    dynamics the sweep parameter is an amplitude factor (smaller factor =
    bigger fault), which is not monotonic with |true_dev| in the same
    direction as e.g. detune cents. Measured deviation is the actual x-axis
    classify() thresholds against, so it is always the right sort key."""
    rows = [r for r in summary_rows if r["mean_abs_true_dev"] is not None]
    rows.sort(key=lambda r: r["mean_abs_true_dev"])
    for r in rows:
        if r["recall"] is not None and r["recall"] >= target:
            return r["mean_abs_true_dev"]
    return None


def run_battery(seed0=100, n_pros=5, base_bpm=66.0, jitter=3.0, fast=False):
    t_start = time.time()
    # one warm-up call: librosa/numba JIT-compiles yin/chroma on first use
    # (~18s), which would otherwise land on whichever trial happens to run
    # first and distort nothing but wall-clock time.
    _ = extract(render(bpm=66, seed=1), sr=SYNTH_SR)

    pros, bpms = build_pros(n=n_pros, base_bpm=base_bpm, jitter=jitter, seed0=seed0)
    corridor = build_corridor(pros)
    ref_bpm = bpms[0]          # build_corridor's timeline is recordings[0]
    mean_bpm = float(np.mean(bpms))

    seeds4 = [900, 901, 902, 903] if not fast else [900]
    seeds3 = [900, 901, 902] if not fast else [900]

    report = {"meta": {"n_pros": n_pros, "pro_bpms": bpms, "ref_bpm": ref_bpm,
                        "mean_bpm": mean_bpm, "phrase_notes": N}}

    print("[1/7] intonation sweep...")
    intonation_vals = [-100, -80, -60, -45, -32, -22, -14, -8, -3, 3, 8, 14, 22, 32, 45, 60, 80, 100]
    rows = sweep_corridor_category("intonation", 4, "detune", intonation_vals, seeds4,
                                    corridor, pros, ref_bpm)
    report["intonation"] = {"rows": rows, "summary": summarize_sweep(rows)}

    print("[2/7] dynamics sweep (softer @ note 4, louder @ note 0)...")
    soft_db = [1, 2, 3, 4, 5, 6, 8, 10, 13, 17]
    soft_vals = [round(10 ** (-d / 20.0), 4) for d in soft_db]
    rows_soft = sweep_corridor_category("dynamics", 4, "dyn", soft_vals, seeds4,
                                         corridor, pros, ref_bpm)
    loud_db = [2, 4, 6, 9, 13]
    loud_vals = [round(10 ** (d / 20.0), 4) for d in loud_db]
    rows_loud = sweep_corridor_category("dynamics", 0, "dyn", loud_vals, seeds3,
                                         corridor, pros, ref_bpm)
    report["dynamics"] = {"rows": rows_soft + rows_loud,
                           "summary_softer": summarize_sweep(rows_soft),
                           "summary_louder": summarize_sweep(rows_loud)}

    print("[3/7] tone (brightness) sweep...")
    bright_vals = [-0.8, -0.6, -0.4, -0.25, -0.12, 0.12, 0.25, 0.4, 0.7, 1.1, 1.8, 3.0]
    rows = sweep_corridor_category("tone", 4, "bright", bright_vals, seeds4,
                                    corridor, pros, ref_bpm)
    report["tone"] = {"rows": rows, "summary": summarize_sweep(rows)}

    print("[4/7] timbre (breathiness) sweep...")
    breath_vals = [0.0004, 0.0008, 0.0015, 0.0025, 0.004, 0.006, 0.009,
                   0.013, 0.02, 0.035, 0.05, 0.08, 0.12, 0.18, 0.28, 0.42]
    rows = sweep_corridor_category("timbre", 4, "breath", breath_vals, seeds4,
                                    corridor, pros, ref_bpm)
    report["timbre"] = {"rows": rows, "summary": summarize_sweep(rows)}

    print("[5/7] tempo sweep...")
    bpm_vals = [40, 48, 56, 64, 66, 70, 76, 84, 94, 108, 126, 150, 180, 210]
    rows = sweep_tempo(bpm_vals, seeds3, corridor, pros, mean_bpm)
    errs = [r["rel_error_pct"] for r in rows]
    report["tempo"] = {"rows": rows, "mean_rel_error_pct": float(np.mean(errs)),
                        "median_rel_error_pct": float(np.median(errs)),
                        "max_rel_error_pct": float(np.max(errs))}

    print("[6/7] vibrato sweep...")
    vib_factors = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.3, 1.6, 2.0, 2.6, 3.5]
    rows = sweep_vibrato(vib_factors, seeds3, corridor, pros, ref_bpm)
    diffs = [abs(r["measured_extent_cents"] - r["true_extent_cents"]) for r in rows
             if np.isfinite(r["measured_extent_cents"])]
    report["vibrato"] = {"rows": rows,
                          "extent_mae_cents": float(np.mean(diffs)) if diffs else None}

    print("[7/7] false-positive sweep on clean playing...")
    n_clean = 12 if fast else 30
    rows = sweep_clean(n_clean, corridor, pros, base_bpm, jitter, seed0=700)
    any_finding = sum(1 for r in rows if r["n_findings"] > 0)
    per_cat = {}
    for r in rows:
        for cat, cnt in r["by_category"].items():
            per_cat[cat] = per_cat.get(cat, 0) + 1
    vib_fp = sum(1 for r in rows if r["vibrato_msgs"])
    report["clean"] = {"rows": rows, "n_trials": n_clean,
                        "trial_fp_rate": any_finding / n_clean,
                        "per_category_fp_rate": {k: v / n_clean for k, v in per_cat.items()},
                        "vibrato_fp_rate": vib_fp / n_clean}

    print("[+] false-positive rate vs. number of reference recordings...")
    report["fp_by_n_pros"] = sweep_fp_by_n_pros([2, 3, 5, 8], 20 if not fast else 8,
                                                 base_bpm=base_bpm, jitter=jitter,
                                                 pro_seed0=seed0, clean_seed0=750)

    # thresholds actually implemented vs. thresholds the data says are needed
    report["floor_check"] = {}
    for cat in ("intonation", "dynamics", "tone", "timbre"):
        key = KEY_OF[cat]
        rows_ = report[cat]["rows"] if cat != "dynamics" else rows_soft
        summ = summarize_sweep(rows_)
        report["floor_check"][cat] = {
            "coded_floor": CL.FLOORS[key],
            "empirical_50pct_recall_at": threshold_crossing(summ, 0.5),
            "empirical_90pct_recall_at": threshold_crossing(summ, 0.9),
        }

    report["meta"]["elapsed_s"] = round(time.time() - t_start, 1)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "accuracy_report.json"))
    ap.add_argument("--fast", action="store_true", help="fewer seeds/trials, for a quick smoke run")
    args = ap.parse_args()

    report = run_battery(fast=args.fast)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\nwrote", args.out, " (%.1fs)" % report["meta"]["elapsed_s"])

    print("\n=== floor check (coded FLOORS vs. what the data needs) ===")
    for cat, d in report["floor_check"].items():
        print("  %-11s coded=%.1f  50%%-recall at |dev|~=%s  90%%-recall at |dev|~=%s"
              % (cat, d["coded_floor"],
                 "%.1f" % d["empirical_50pct_recall_at"] if d["empirical_50pct_recall_at"] else "n/a",
                 "%.1f" % d["empirical_90pct_recall_at"] if d["empirical_90pct_recall_at"] else "n/a"))

    print("\n=== tempo ===")
    t = report["tempo"]
    print("  mean rel error: %.2f%%  median: %.2f%%  max: %.2f%%"
          % (t["mean_rel_error_pct"], t["median_rel_error_pct"], t["max_rel_error_pct"]))

    print("\n=== false positives on clean playing (n=%d) ===" % report["clean"]["n_trials"])
    print("  any spurious finding: %.0f%% of trials" % (100 * report["clean"]["trial_fp_rate"]))
    for cat, rate in report["clean"]["per_category_fp_rate"].items():
        print("    %-11s %.0f%%" % (cat, 100 * rate))
    print("  vibrato message fired: %.0f%% of trials" % (100 * report["clean"]["vibrato_fp_rate"]))

    print("\n=== vibrato extent tracking ===")
    print("  mean abs error vs. true injected extent: %.1f cents"
          % report["vibrato"]["extent_mae_cents"])

    print("\n=== false-positive rate vs. number of reference recordings ===")
    for k, rate in sorted(report["fp_by_n_pros"].items()):
        print("  %d pros: %.0f%% of clean trials flagged" % (k, 100 * rate))


if __name__ == "__main__":
    main()

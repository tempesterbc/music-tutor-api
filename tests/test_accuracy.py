"""
test_accuracy.py - fast pytest regression checks for the analysis engine's
detection accuracy.

These are a small, fixed subset of what `accuracy_battery.py` measures in
full: enough to catch a real regression (a detector going blind, or firing
on everything) in well under a minute, without re-running the full sweep.
For the actual accuracy numbers - recall curves, false-positive rates,
tempo error, floor calibration - run:

    python tests/accuracy_battery.py

and see tests/README.md.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_synth import render, SR as SYNTH_SR                    # noqa: E402
from analysis_core import extract, build_corridor                # noqa: E402
import classify as CL                                            # noqa: E402
from tests.accuracy_battery import (                             # noqa: E402
    build_pros, render_extract, faulted, ref_frame_range,
    measured_deviation, check_detected, sweep_clean, sweep_fp_by_n_pros,
)

FAULT_NOTE = 4


@pytest.fixture(scope="module")
def corridor_and_pros():
    # pay the one-time librosa/numba JIT cost once for the whole module
    extract(render(bpm=66, seed=1), sr=SYNTH_SR)
    pros, bpms = build_pros(n=5, base_bpm=66.0, jitter=3.0, seed0=100)
    corridor = build_corridor(pros)
    return corridor, pros, bpms[0]


def _trial(corridor, pros, ref_bpm, cat, key, note_idx, **fault):
    student = render_extract(bpm=ref_bpm, seed=1, **faulted(note_idx, **fault))
    al = CL.align(student, corridor)
    findings, _ = CL.classify(student, corridor, pros, alignment=al)
    a, b = ref_frame_range(corridor, ref_bpm, note_idx)
    dev = measured_deviation(corridor, al, key, a, b)
    hit = check_detected(findings, cat, a, b, corridor["n"], dev)
    return dev, hit


@pytest.mark.parametrize("delta_cents", [60, -60])
def test_intonation_detects_large_fault(corridor_and_pros, delta_cents):
    corridor, pros, ref_bpm = corridor_and_pros
    dev, hit = _trial(corridor, pros, ref_bpm, "intonation", "cents", FAULT_NOTE,
                       detune=delta_cents)
    assert hit is not None, "a %+d cent fault (3x the 20-cent floor) should be flagged" % delta_cents
    assert hit[1] > 0.4, "flagged region should substantially overlap the faulted note"


def test_intonation_ignores_small_fault(corridor_and_pros):
    corridor, pros, ref_bpm = corridor_and_pros
    dev, hit = _trial(corridor, pros, ref_bpm, "intonation", "cents", FAULT_NOTE, detune=3)
    assert hit is None, "a 3-cent wobble should stay inside the professional corridor"


def test_dynamics_detects_large_fault(corridor_and_pros):
    corridor, pros, ref_bpm = corridor_and_pros
    dev, hit = _trial(corridor, pros, ref_bpm, "dynamics", "db", FAULT_NOTE, dyn=0.2)
    assert hit is not None


def test_tone_detects_large_fault(corridor_and_pros):
    corridor, pros, ref_bpm = corridor_and_pros
    dev, hit = _trial(corridor, pros, ref_bpm, "tone", "centroid", FAULT_NOTE, bright=2.0)
    assert hit is not None


def test_timbre_detects_large_fault(corridor_and_pros):
    corridor, pros, ref_bpm = corridor_and_pros
    dev, hit = _trial(corridor, pros, ref_bpm, "timbre", "flatness", FAULT_NOTE, breath=0.15)
    assert hit is not None


@pytest.mark.parametrize("bpm", [50, 66, 99])
def test_tempo_ratio_within_tolerance(corridor_and_pros, bpm):
    corridor, pros, ref_bpm = corridor_and_pros
    student = render_extract(bpm=bpm, seed=1)
    al = CL.align(student, corridor)
    _, summary = CL.classify(student, corridor, pros, alignment=al)
    expected = ref_bpm / bpm
    rel_err = abs(summary["tempo_ratio"] - expected) / expected
    assert rel_err < 0.15, (
        "tempo ratio off by %.0f%% at %d bpm (expected ~%.2f, got %.2f)"
        % (100 * rel_err, bpm, expected, summary["tempo_ratio"]))


def test_clean_playing_false_positive_rate_bounded(corridor_and_pros):
    """Not a target of zero: FLOORS is deliberately tight for intonation/tone,
    and dynamics/timbre run mostly on 2x professional std (see classify.py's
    module docstring and tests/README.md). This guards against a REGRESSION
    (e.g. every clean trial suddenly getting flagged), not against the
    already-known, already-documented baseline rate."""
    corridor, pros, ref_bpm = corridor_and_pros
    rows = sweep_clean(10, corridor, pros, base_bpm=66.0, jitter=3.0, seed0=700)
    fp_rate = sum(1 for r in rows if r["n_findings"] > 0) / len(rows)
    assert fp_rate < 0.75, "false-positive rate on clean playing regressed sharply (%.0f%%)" % (100 * fp_rate)


def test_more_reference_recordings_does_not_increase_false_positives(corridor_and_pros):
    """Sanity check on build_corridor's ddof=1 sample-std fix: going from 2 to
    8 references should never make the corridor MORE trigger-happy."""
    rates = sweep_fp_by_n_pros([2, 8], n_clean=8, clean_seed0=750)
    assert rates[8] <= rates[2] + 0.15

# Engine accuracy validation — 28 Aug 2026

Quantitative accuracy numbers for the analysis engine (`analysis_core.py` +
`classify.py`), measured by `tests/accuracy_battery.py` against synthetic
recordings with known, controlled faults. See `tests/README.md` for why
synthetic ground truth (not real recordings) is what produces a real
recall/precision curve, and what it would take to add real audio.

Setup: one 5-recording professional corridor (`perf_synth.py`, bpm ~63-69,
seeds 100-104), the same corridor used for every measurement below unless
stated otherwise. Full raw output: `tests/accuracy_report.json` (regenerate
with `python tests/accuracy_battery.py`).

## Detection accuracy by category

For each category, one note (2 beats, mid-phrase) was rendered with a known
deviation and swept from below to well above the coded `FLOORS` value in
`classify.py`. "50%/90%-recall point" is the smallest *measured* deviation
(computed the same way `classify.py` computes it internally, not the raw
synth-parameter value, which isn't linear in the descriptor) at which the
engine detected it in that fraction of trials (4 seeds/level, IoU > 0.15
against the true note span required to count as a hit).

| category | coded floor | 50%-recall at | 90%-recall at | median IoU on hit |
|---|---|---|---|---|
| intonation | 20 cents | 21.7 cents | 34.5 cents | 0.94 |
| tone (brightness) | 2.4 semitones | 2.4 st | 3.2 st | 0.94–0.98 |
| dynamics | 4.0 dB | 5.3 dB | 5.3 dB | 0.82–0.92 |
| timbre (breathiness) | 4.5 dB | 6.0 dB | 8.8 dB | 0.85–0.92 |

Intonation and tone track their coded floors closely — the floor is, in
practice, close to the real detection threshold, and once something is
flagged the reported time span tightly bounds the actual faulted note
(IoU regularly above 0.9). Dynamics and timbre need roughly 1.3–2x their
floor before they reliably fire; that's consistent with the 26 Aug review's
note that "only dynamics and timbre are meaningfully driven by professional
variance" — for those two, `2 × pro_std` is usually the binding term, not
the floor, so the effective threshold sits above the floor's nominal value.

Both detection curves were clean step functions (0% recall well below the
threshold, 100% well above), not gradual — there's a real, sharp boundary
here, not just noisy behavior.

## Tempo

Swept 40–210 bpm (against a ~66 bpm mean corridor), 3 seeds/tempo:

- Median relative error: 2.5%; mean 2.8%.
- Error stays under ~4% across the entire 48–150 bpm range.
- It rises at the extremes: 48 bpm (1.4x slower than the corridor) hit 7.5%
  on one seed, and 210 bpm (3.2x faster) ran 5.6–10.3%. That tracks the DTW's
  Sakoe–Chiba band (`band_rad=0.25`): a tempo ratio this far from 1.0 pushes
  the alignment path toward the edge of what the band allows it to warp.

Practical read: tempo accuracy is strong for anything in the range a real
student would plausibly play at (even quite rushed or dragging), and
degrades gracefully — not catastrophically — only at tempo ratios well
outside normal practice-room variation.

## False positives on clean, fault-free playing

30 synthetic "clean" takes (same distribution as the professional
references, different seed only — by construction there is no true fault in
any of them) were run against the corridor:

- 33% of clean trials tripped at least one spurious finding.
- Every spurious finding was `dynamics: softer` or `timbre: breathier/airier`
  — never intonation, tone, or the opposite direction. Magnitudes were
  5–18 dB, well above the coded floors, so these aren't borderline
  rounding — the engine is confidently, incorrectly flagging real playing.
- **This does not improve with more reference recordings.** Rebuilding the
  corridor from 2, 3, 5, and 8 references (same 20 clean test takes each
  time) gave 30% false-positive rate at every single corridor size.

That last point matters: it rules out "not enough references yet" as the
explanation (which is what the 26 Aug review's ddof fix targeted). The floor
itself is tight enough, relative to ordinary take-to-take loudness/timbre
variation in this synthesis pipeline, that clean playing routinely crosses
it regardless of how well-estimated the corridor's std is. Two candidate
fixes, not yet tried: raise `FLOORS["db"]`/`FLOORS["flatness"]`, or address
the underlying source of the variation (each clip is normalised to its own
peak sample — `y = y / (np.max(np.abs(y)) + eps)` in `perf_synth.render` —
which can shift a whole clip's loudness a few dB based on one random
transient; per-note or RMS-based normalisation would likely be steadier).

## Vibrato: currently cannot fire, on any input

This is the most significant finding. Swept vibrato extent from 0 to 3.5x
the professional baseline (0 to 49 cents, vs. the 14-cent baseline every
reference recording also uses):

- **The "present vibrato" gate in `descriptors.vibrato()` never once passed
  — not for the synthetic professional references, and not for any student
  trial, at any factor from 0 to 3.5x.** `vibrato_profile()`'s aggregate
  `extent` was `NaN` for every one of the 5 corridor references. Since
  `classify.py`'s vibrato section only emits a message when the professional
  median extent list (`pe`) is non-empty, and it is *always* empty, the
  vibrato message cannot fire for any student, regardless of how much or how
  little vibrato they actually play. This is stronger than "uncalibrated" —
  the feature is currently dead code in practice.
- Root cause, most likely: `descriptors.vibrato()` requires a measured
  extent above 8 cents to count a note as "present." But the *measured*
  extent on a note synthesised with a true 14-cent vibrato came out to
  0.1–4 cents — roughly 5–15% of the true value. `analysis_core.extract()`
  runs a 5-frame rolling median (`_roll_median(cents, 5)`, ≈0.16 s at
  perf_synth's 5.5 Hz vibrato rate ≈ 0.18 s/cycle) over the pitch curve to
  kill isolated octave errors. That window is almost exactly one vibrato
  cycle, which would flatten a genuine vibrato oscillation down to a small
  fraction of its real size — consistent with what was measured. (This is
  the most likely mechanism, not a fully isolated proof — it wasn't tested
  by bypassing the smoothing directly.)
- At very wide, exaggerated vibrato (2.0–3.5x baseline) a handful of notes
  did register, but still badly undercounted the true extent (measured
  8–13 cents against a true 28–49 cents).

Practical read: don't trust the vibrato message on real audio yet — not
because it's "uncalibrated" in the sense of needing different thresholds,
but because the pitch-smoothing step upstream of it removes most of the
signal it needs before it ever reaches the present/absent gate. Lowering
the 8-cent gate would not fix this by itself; the extent estimate itself is
suppressed at the source.

## Summary for the writeup

| detector | verdict |
|---|---|
| Tempo (Theil–Sen DTW slope) | Accurate within ~3% across normal tempo range, degrades gracefully at extremes |
| Intonation | Well-calibrated to its stated floor, tight localisation |
| Tone (brightness) | Well-calibrated to its stated floor, tight localisation |
| Dynamics | Functional but needs ~1.3x its stated floor to reliably fire |
| Timbre (breathiness) | Functional but needs ~1.3–2x its stated floor to reliably fire |
| Dynamics / timbre specificity | ~1/3 of demonstrably clean playing gets a spurious finding, independent of reference-recording count |
| Vibrato | Currently non-functional — the present/absent gate never passes, on references or students, at any tested extent |

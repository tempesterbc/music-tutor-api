# Accuracy validation

Two files here answer "how accurate is the engine": `accuracy_battery.py` (the
measurement) and `test_accuracy.py` (fast pytest regression guards built on
top of it).

## Why synthetic recordings, not real ones

Real recordings can sanity-check that the engine behaves sensibly, but they
cannot produce a recall/precision curve, because there is no ground truth for
"how many cents flat was that note" or "how many dB softer" in a real
recording - only opinions. `perf_synth.py` exists for exactly this reason
(see its own docstring): it synthesises a phrase with per-note controls
(detune in cents, amplitude, brightness, breathiness, vibrato extent,
timing), so every "fault" has a known, exact size. That is what
`accuracy_battery.py` uses to measure detection recall as a function of true
fault size, not just "did it work on this one clip."

We looked at getting real annotated audio too (checked URMP - University of
Rochester Multi-modal Music Performance dataset - and Bach10, both used in
MIR research for exactly this kind of task). URMP has individual
solo-instrument tracks with note-level pitch/onset annotations across 44
chamber pieces, which is the right shape for this project, but it is a 12.5
GB download gated behind a request form, and its redistribution terms aren't
published on the project page - not something to pull into a public repo's
test suite sight unseen. If real audio is wanted for a qualitative check
later, the cheapest, cleanest path is Bowen's own recordings (playing an
excerpt cleanly, then again with a deliberate flaw) - real ground truth,
zero licensing questions, and it tests the thing the app is actually for.

## Running it

```
pip install -r requirements.txt
python tests/accuracy_battery.py            # full battery, ~2 min, writes tests/accuracy_report.json
python tests/accuracy_battery.py --fast     # smoke test, ~30s, fewer seeds
python -m pytest tests/test_accuracy.py -v  # fast regression checks, ~15s
```

`accuracy_battery.py` builds one 5-recording professional corridor, then for
each of tempo / intonation / dynamics / tone / timbre / vibrato: renders
students with a controlled fault at a known note and severity, aligns them,
classifies them, and checks whether the reported finding (a) exists, (b)
has the right sign/label, and (c) overlaps the true faulted note (IoU). It
also measures the false-positive rate on demonstrably fault-free playing,
and how that rate changes with the number of reference recordings.

Findings from the last full run are written up in
`docs/accuracy-validation-2026-08-28.md`. Re-run the battery before trusting
those numbers after any change to `classify.py`, `analysis_core.py`, or
`descriptors.py` - they are measurements of a specific commit, not a
promise about all future ones.

"""
perf_synth.py - expressive STAND-IN recordings (no copyrighted audio).

Synthesises a short original phrase with controllable, per-note "problems" so we
can prove the classifier detects each one. Replace with real audio to analyse a
real piece - the analysis code is format-agnostic.

Per-note controls (all length-N lists, defaults = clean professional playing):
  dyn        amplitude 0..1
  detune     intonation offset in cents
  bright     upper-harmonic boost (tone: + brighter/harsher, - duller)
  breath     additive noise level (timbre: breathy/airy)
  vext       vibrato extent in cents
  vjit       vibrato rate irregularity 0..1 (shaky vibrato)
  toff       onset time offset in seconds (timing / evenness)
"""
import numpy as np

SR = 22050
PHRASE = [
    (69, 1.0), (71, 1.0), (72, 1.5), (74, 0.5), (76, 2.0),
    (74, 1.0), (72, 1.0), (71, 1.5), (69, 0.5), (67, 2.0),
    (69, 1.0), (72, 3.0),
]


def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def _arr(v, N, default):
    if v is None:
        return np.full(N, default, float)
    return np.asarray(v, float)


def render(bpm=66, dyn=None, detune=None, bright=None, breath=None,
           vext=None, vjit=None, toff=None, vrate=5.5, seed=0,
           onset_jitter_ms=7.0, base_noise=0.0008, sr=SR):
    rng = np.random.default_rng(seed)
    N = len(PHRASE)
    arch = np.sin(np.linspace(0.15, np.pi - 0.15, N))
    dyn = _arr(dyn, N, None) if dyn is not None else 0.5 + 0.5 * arch
    detune = _arr(detune, N, 0.0) if detune is not None else rng.normal(0, 4, N)
    bright = _arr(bright, N, 0.0)
    breath = _arr(breath, N, 0.0)
    vext = _arr(vext, N, 14.0)
    vjit = _arr(vjit, N, 0.06)
    toff = _arr(toff, N, 0.0)

    beat = 60.0 / bpm
    starts, t = [], 0.0
    for i, (_, b) in enumerate(PHRASE):
        starts.append(t + toff[i] + rng.normal(0, onset_jitter_ms / 1000.0))
        t += b * beat
    y = np.zeros(int((t + 2.0) * sr) + 1)

    for i, (midi, b) in enumerate(PHRASE):
        dur = b * beat * 0.92
        ns = int(dur * sr)
        tt = np.arange(ns) / sr
        # vibrato with (optional) irregular rate -> "shaky"
        rate_walk = np.cumsum(rng.normal(0, vjit[i], ns)) * 0.5
        rate_inst = vrate * (1.0 + 0.15 * np.tanh(rate_walk))
        theta = 2 * np.pi * np.cumsum(rate_inst) / sr
        vib = vext[i] * np.sin(theta + rng.uniform(0, 6.28))
        freq = midi_to_hz(midi) * 2.0 ** ((detune[i] + vib) / 1200.0)
        phi = 2 * np.pi * np.cumsum(freq) / sr
        wave = np.zeros(ns)
        for h in range(1, 8):
            gain = (1.0 / h) * (1.0 + bright[i] * (h - 1) / 6.0)
            wave += gain * np.sin(h * phi)
        wave += breath[i] * rng.standard_normal(ns)      # breathiness
        env = np.ones(ns)
        a, d = int(0.02 * sr), int(0.08 * sr)
        if a: env[:a] = np.linspace(0, 1, a)
        if d: env[-d:] = np.linspace(1, 0, d)
        wave *= env * dyn[i]
        i0 = max(0, int(starts[i] * sr))
        y[i0:i0 + ns] += wave

    y = y / (np.max(np.abs(y)) + 1e-9)
    y = y + base_noise * rng.standard_normal(len(y))
    return y.astype(np.float32)

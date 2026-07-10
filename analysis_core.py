"""
analysis_core.py - score-free comparison of performances against a consensus of
professional recordings.

For every recording we extract time series:
    chroma     - for robust DTW alignment between recordings
    cents      - intonation (pitch relative to A440)
    db         - loudness / dynamics (relative)
    centroid   - spectral centroid in semitone-ish log-Hz (tone brightness)
    flatness   - spectral flatness in dB (breathy/noisy vs pure tone)

DTW warps every professional onto a shared timeline (the first pro). We fuse
them into a "professional corridor": mean +/- std at each point for each feature,
plus a professional TIMING curve (used to spot local rushing/dragging/unevenness).
A student is then aligned onto the same timeline and compared feature by feature.
"""
import warnings
import numpy as np
import librosa
warnings.filterwarnings("ignore", category=RuntimeWarning)

SR = 16000
HOP = 512
A4 = 440.0
FRAME_RATE = SR / HOP   # ~43.07 feature frames per second


def load_audio(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def _roll_median(x, k=5):
    """NaN-aware rolling median - removes isolated octave errors from yin."""
    out = np.full_like(x, np.nan)
    h = k // 2
    for i in range(len(x)):
        seg = x[max(0, i - h): i + h + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def extract(path_or_y, sr=SR, max_seconds=45):
    """Return dict of per-frame features. Uses the FAST path (yin + chroma_stft
    at 16 kHz) so it runs in a few seconds even on a small CPU."""
    if isinstance(path_or_y, str):
        y = load_audio(path_or_y, sr=sr)
    else:
        y = np.asarray(path_or_y, dtype=float)
    if len(y) > max_seconds * sr:            # cap very long uploads
        y = y[: max_seconds * sr]

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)
    chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)

    # fast pitch: yin (no probabilistic matrix); gate voicing by energy
    f0 = librosa.yin(y, sr=sr, fmin=float(librosa.note_to_hz("C2")),
                     fmax=float(librosa.note_to_hz("C7")),
                     frame_length=2048, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    n = min(len(f0), len(rms), chroma.shape[1])
    f0, rms, chroma = f0[:n], rms[:n], chroma[:, :n]
    voiced = rms > (0.06 * (rms.max() + 1e-9))
    cents = np.full(n, np.nan)
    ok = voiced & (f0 > 0)
    cents[ok] = 1200.0 * np.log2(f0[ok] / A4)
    cents = _roll_median(cents, 5)           # kill isolated octave jumps
    # octave-error correction: snap each note to the octave nearest the local
    # melodic contour (yin sometimes reports f0*2 or f0/2 on real audio)
    ref = _roll_median(cents, 15)
    for i in range(len(cents)):
        if not np.isnan(cents[i]) and not np.isnan(ref[i]):
            while cents[i] - ref[i] > 600:
                cents[i] -= 1200
            while cents[i] - ref[i] < -600:
                cents[i] += 1200

    db = 20.0 * np.log10(rms + 1e-6)
    db = db - np.nanmedian(db)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0][:n]
    centroid = 12.0 * np.log2(np.maximum(centroid, 1e-6) / A4)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=HOP)[0][:n]
    flatness = 10.0 * np.log10(flatness + 1e-9)
    centroid = np.where(np.isnan(cents), np.nan, centroid)
    flatness = np.where(np.isnan(cents), np.nan, flatness)

    onset_f = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP,
                                         backtrack=False, wait=8, delta=0.03)
    T = min(chroma.shape[1], len(cents), len(db), len(centroid), len(flatness))
    times = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=HOP)
    onset_f = onset_f[onset_f < T]
    return {"chroma": chroma[:, :T], "cents": cents[:T], "db": db[:T],
            "centroid": centroid[:T], "flatness": flatness[:T],
            "onset_f": onset_f, "times": times}


def warp_map(ref_chroma, other_chroma):
    """DTW map: reference-frame index -> list of matched other-frame indices."""
    # +eps so silent (all-zero) frames do not make cosine distance NaN
    ref_chroma = np.nan_to_num(ref_chroma) + 1e-6
    other_chroma = np.nan_to_num(other_chroma) + 1e-6
    _, wp = librosa.sequence.dtw(X=ref_chroma, Y=other_chroma, metric="cosine")
    wp = wp[::-1]
    m = {}
    for r, o in wp:
        m.setdefault(int(r), []).append(int(o))
    return m


def warp_curve(curve, ref_to_other, n_ref):
    out = np.full(n_ref, np.nan)
    for r in range(n_ref):
        idx = ref_to_other.get(r, [])
        if idx:
            vals = curve[idx]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                out[r] = np.median(vals)
    return out


def warp_times(other_times, ref_to_other, n_ref):
    """For each reference frame, the matched TIME in the other recording."""
    out = np.full(n_ref, np.nan)
    for r in range(n_ref):
        idx = ref_to_other.get(r, [])
        if idx:
            out[r] = np.median([other_times[i] for i in idx])
    return out


def _smooth(x, k=9):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    h = k // 2
    for i in range(len(x)):
        seg = x[max(0, i - h): i + h + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = np.mean(seg)
    return out


def build_corridor(recordings):
    """recordings: list of extract() dicts. First defines the timeline."""
    ref = recordings[0]
    n = ref["chroma"].shape[1]
    stacks = {k: [] for k in ("cents", "db", "centroid", "flatness")}
    time_stack = []
    for rec in recordings:
        m = warp_map(ref["chroma"], rec["chroma"])
        for k in stacks:
            stacks[k].append(warp_curve(rec[k], m, n))
        time_stack.append(warp_times(rec["times"], m, n))

    cents_stack = np.vstack(stacks["cents"])
    coverage = np.mean(~np.isnan(cents_stack), axis=0)
    valid = coverage >= 0.6
    valid[:max(1, int(0.02 * n))] = False
    valid[int(0.96 * n):] = False

    corridor = {"n": n, "times": ref["times"], "valid": valid,
                "ref_chroma": ref["chroma"], "frame_rate": FRAME_RATE,
                "ref_onsets": ref["onset_f"]}
    for k, st in stacks.items():
        M = np.vstack(st)
        corridor[k + "_mean"] = _smooth(np.nanmean(M, axis=0))
        corridor[k + "_std"] = _smooth(np.nanstd(M, axis=0))
    corridor["cents_mean"][~valid] = np.nan

    # professional timing curve + how much pros disagree on local timing
    time_stack = np.vstack(time_stack)
    corridor["time_mean"] = np.nanmean(time_stack, axis=0)
    corridor["time_std"] = np.nanstd(time_stack, axis=0)
    return corridor


def segment_reference(corridor, min_frames=8):
    """Note regions on the reference timeline, bounded by the reference's own
    onsets. Robust: one region per played note, so analysis is per-note."""
    onsets = list(corridor["ref_onsets"])
    n = corridor["n"]
    valid = corridor["valid"]
    bounds = [o for o in onsets if 0 <= o < n]
    if not bounds or bounds[0] > 2:
        bounds = [0] + bounds
    bounds = sorted(set(bounds)) + [n]
    regions = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        b = b - 1
        # trim to the valid, voiced interior
        if b - a + 1 >= min_frames and np.any(valid[a:b + 1]):
            regions.append((a, b))
    return regions

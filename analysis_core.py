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
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore", category=RuntimeWarning)

SR = 16000
HOP = 512
A4 = 440.0
FRAME_RATE = SR / HOP   # ~31.25 feature frames per second
MAX_SECONDS = 45        # analysis window cap (keeps a request bounded)

# Sakoe-Chiba band for the DTW: the alignment may deviate from the straight
# diagonal by at most this fraction of the longer recording. Two jobs:
#  - stops the alignment taking a cheap detour through a repeated phrase, which
#    was manufacturing phantom "rushing here" findings on clean playing;
#  - bounds the cost matrix, so a long upload can't blow the memory budget.
DTW_BAND_RAD = 0.25


class UnreadableAudio(Exception):
    """Raised when a file cannot be decoded, so callers can answer 400 rather
    than surfacing a libsndfile traceback as a 500."""


def load_audio(path, sr=SR):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        raise UnreadableAudio(str(e) or type(e).__name__) from e
    if y is None or len(y) < sr // 10:
        raise UnreadableAudio("the file contains less than 0.1 s of audio")
    return y


def _windowed(x, k):
    """Return an (len(x), k) view of x with NaN padding at both edges."""
    h = k // 2
    pad = np.full(h, np.nan)
    return sliding_window_view(np.concatenate([pad, np.asarray(x, float), pad]), k)


def _roll_median(x, k=5):
    """NaN-aware rolling median - removes isolated octave errors from yin.
    Vectorised: same result as the per-frame loop, without the Python overhead."""
    return np.nanmedian(_windowed(x, k), axis=1)


def _smooth(x, k=9):
    """NaN-aware rolling mean."""
    return np.nanmean(_windowed(x, k), axis=1)


def extract(path_or_y, sr=SR, max_seconds=MAX_SECONDS):
    """Return dict of per-frame features. Uses the FAST path (yin + chroma_stft
    at 16 kHz) so it runs in a few seconds even on a small CPU.

    `sr` is the sample rate of `path_or_y` when an array is passed; audio at a
    different rate is resampled first, so the reported times are always real
    seconds. (Passing 22 kHz audio while assuming 16 kHz used to stretch every
    reported timestamp by ~38%.)
    """
    if isinstance(path_or_y, str):
        y = load_audio(path_or_y, sr=SR)
    else:
        y = np.asarray(path_or_y, dtype=float)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    sr = SR

    full_seconds = len(y) / sr                     # true length, before capping
    if len(y) > max_seconds * sr:                  # cap very long uploads
        y = y[: int(max_seconds * sr)]

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
    shift = np.where(np.isfinite(cents - ref), np.round((cents - ref) / 1200.0), 0.0)
    cents = cents - 1200.0 * shift

    db = 20.0 * np.log10(rms + 1e-6)
    # Reference the loudness curve to the VOICED frames only. Using the median of
    # every frame let leading/trailing silence drag the whole curve up or down,
    # which showed up later as a phantom "too loud"/"too soft" finding.
    ref_db = db[voiced] if voiced.any() else db
    db = db - np.median(ref_db)

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
            "onset_f": onset_f, "times": times,
            "full_seconds": full_seconds, "truncated": full_seconds > max_seconds}


def warp_map(ref_chroma, other_chroma):
    """DTW map: reference-frame index -> list of matched other-frame indices."""
    # +eps so silent (all-zero) frames do not make cosine distance NaN
    ref_chroma = np.nan_to_num(ref_chroma) + 1e-6
    other_chroma = np.nan_to_num(other_chroma) + 1e-6
    _, wp = librosa.sequence.dtw(X=ref_chroma, Y=other_chroma, metric="cosine",
                                 global_constraints=True, band_rad=DTW_BAND_RAD)
    wp = wp[::-1]
    m = {}
    for r, o in wp:
        m.setdefault(int(r), []).append(int(o))
    return m


def identity_map(n):
    """The warp of a recording onto itself - used for the corridor's reference
    so we don't pay for a DTW that is known in advance to be the diagonal."""
    return {i: [i] for i in range(n)}


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
            out[r] = np.median(other_times[idx])
    return out


def local_tempo_ratio(ref_times, matched_times, valid=None, win_s=3.0, smooth_s=1.0):
    """NOT currently used for reporting - see the note in classify.py. Kept for
    the score-based path, where exact expected onsets make it meaningful.

    Local slope d(performer time)/d(reference time), one value per reference
    frame, from a centred window of `win_s` seconds.

    >1 means the performer is spending MORE time than the reference here
    (dragging); <1 means rushing. Returns NaN where there isn't enough data.

    The DTW path is a staircase - it holds one frame flat, then jumps - so the
    raw slope swings between 0 and several hundred percent frame to frame. The
    matched-time curve is smoothed first (`smooth_s`) and the regression window
    is wide, otherwise the "rushing here" detector fires on staircase noise
    rather than on playing.
    """
    matched_times = _smooth(matched_times, max(3, int(smooth_s * FRAME_RATE) | 1))
    k = max(5, int(win_s * FRAME_RATE) | 1)
    xs, ys = _windowed(ref_times, k), _windowed(matched_times, k)
    if valid is not None:
        ys = np.where(_windowed(valid.astype(float), k) > 0, ys, np.nan)
    good = np.isfinite(xs) & np.isfinite(ys)
    cnt = good.sum(axis=1)
    xs = np.where(good, xs, np.nan)
    ys = np.where(good, ys, np.nan)
    mx = np.nanmean(xs, axis=1, keepdims=True)
    my = np.nanmean(ys, axis=1, keepdims=True)
    sxy = np.nansum((xs - mx) * (ys - my), axis=1)
    sxx = np.nansum((xs - mx) ** 2, axis=1)
    slope = np.where((cnt >= 5) & (sxx > 1e-9), sxy / np.where(sxx > 0, sxx, 1), np.nan)
    return slope


def overall_tempo_ratio(ref_times, matched_times, valid=None):
    """Whole-performance tempo ratio, measured from the DTW alignment rather than
    from raw clip length.

    Clip length is the wrong measure twice over: trailing silence counts as
    "slower", and once both recordings are longer than the analysis cap they are
    both truncated to the same number of seconds, so the ratio collapses to
    exactly 1.00 no matter how differently they were played.
    """
    x, y = np.asarray(ref_times, float), np.asarray(matched_times, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if valid is not None:
        ok &= valid
    if ok.sum() < 10:
        return float("nan")
    x, y = x[ok], y[ok]
    # robust slope through the alignment path: median of pairwise slopes over a
    # coarse subsample (Theil-Sen), so a few badly matched frames can't tilt it
    idx = np.linspace(0, len(x) - 1, min(len(x), 160)).astype(int)
    xs, ys = x[idx], y[idx]
    dx = xs[None, :] - xs[:, None]
    dy = ys[None, :] - ys[:, None]
    keep = np.abs(dx) > 0.5           # only pairs at least half a second apart
    if not keep.any():
        return float("nan")
    return float(np.median(dy[keep] / dx[keep]))


def build_corridor(recordings):
    """recordings: list of extract() dicts. First defines the timeline."""
    ref = recordings[0]
    n = ref["chroma"].shape[1]
    stacks = {k: [] for k in ("cents", "db", "centroid", "flatness")}
    time_stack = []
    for i, rec in enumerate(recordings):
        # the reference maps onto itself; no need to run DTW for it
        m = identity_map(n) if rec is ref else warp_map(ref["chroma"], rec["chroma"])
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
                "ref_onsets": ref["onset_f"], "n_pros": len(recordings)}
    # Sample standard deviation (ddof=1). With only 2-3 reference recordings the
    # population std systematically understates how much professionals actually
    # disagree, which narrows the corridor and manufactures findings.
    ddof = 1 if len(recordings) > 1 else 0
    for k, st in stacks.items():
        M = np.vstack(st)
        corridor[k + "_mean"] = _smooth(np.nanmean(M, axis=0))
        corridor[k + "_std"] = _smooth(np.nanstd(M, axis=0, ddof=ddof))
    corridor["cents_mean"][~valid] = np.nan

    # professional timing curve + how much pros disagree on local timing
    time_stack = np.vstack(time_stack)
    corridor["time_mean"] = np.nanmean(time_stack, axis=0)
    corridor["time_std"] = np.nanstd(time_stack, axis=0, ddof=ddof)
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

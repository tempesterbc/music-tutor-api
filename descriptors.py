"""
descriptors.py - note segmentation and vibrato analysis from a pitch (cents) curve.
Used by the classifier to judge vibrato quality and note-level evenness.
"""
import numpy as np


def segment_notes(cents, valid=None, min_dur_frames=13, stab_cents=45.0):
    """Find sustained, stable-pitch, voiced segments (candidate held notes).
    Returns list of (start, end) inclusive frame indices."""
    n = len(cents)
    voiced = ~np.isnan(cents)
    if valid is not None:
        voiced = voiced & valid
    # local stability: rolling std over ~7 frames small
    stable = np.zeros(n, bool)
    for i in range(n):
        seg = cents[max(0, i - 3): i + 4]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= 4 and np.std(seg) < stab_cents:
            stable[i] = True
    ok = voiced & stable
    runs, i = [], 0
    while i < n:
        if ok[i]:
            j = i
            while j < n and ok[j]:
                j += 1
            if j - i >= min_dur_frames:
                runs.append((i + 2, j - 3))   # trim attack/release edges
            i = j
        else:
            i += 1
    return [(a, b) for a, b in runs if b - a >= 6]


def _detrend(x, k):
    """Remove slow trend (moving average) to isolate vibrato oscillation."""
    h = k // 2
    trend = np.array([np.nanmean(x[max(0, i - h): i + h + 1]) for i in range(len(x))])
    return x - trend


def vibrato(cents_seg, frame_rate):
    """Estimate vibrato of one held note. Returns dict with rate (Hz), extent
    (cents), rate_cv (cycle-length variability = 'shakiness'), and present flag."""
    seg = np.asarray(cents_seg, float)
    seg = seg[~np.isnan(seg)]
    if len(seg) < 10:
        return {"present": False, "rate": np.nan, "extent": np.nan, "rate_cv": np.nan}
    trend_win = max(5, int(0.30 * frame_rate))     # ~300 ms
    osc = _detrend(seg, trend_win)
    extent = (np.percentile(osc, 90) - np.percentile(osc, 10)) / 2.0

    # zero-crossings -> half-cycle lengths
    sign = np.sign(osc)
    sign[sign == 0] = 1
    zc = np.where(np.diff(sign) != 0)[0]
    if len(zc) < 4:
        return {"present": False, "rate": np.nan, "extent": float(extent), "rate_cv": np.nan}
    half = np.diff(zc).astype(float)
    rate = frame_rate / (2.0 * np.median(half))
    rate_cv = np.std(half) / (np.mean(half) + 1e-9)
    present = (extent > 8.0) and (3.5 <= rate <= 9.0) and (len(zc) >= 6)
    return {"present": bool(present), "rate": float(rate),
            "extent": float(extent), "rate_cv": float(rate_cv)}


def vibrato_profile(cents, frame_rate, valid=None):
    """Aggregate vibrato stats over all held notes in a recording."""
    segs = segment_notes(cents, valid)
    rates, extents, cvs, present = [], [], [], 0
    for a, b in segs:
        v = vibrato(cents[a:b + 1], frame_rate)
        if v["present"]:
            present += 1
            rates.append(v["rate"]); extents.append(v["extent"]); cvs.append(v["rate_cv"])
    return {"n_notes": len(segs), "n_vib": present,
            "rate": float(np.median(rates)) if rates else np.nan,
            "extent": float(np.median(extents)) if extents else np.nan,
            "rate_cv": float(np.median(cvs)) if cvs else np.nan}

"""
FastAPI backend for the AI Music Practice Tutor.

Wraps the analysis engine (professional-corridor classifier + exercise
recommender) behind a small HTTP API the front-end calls.

Endpoints
  GET  /api/health         - liveness check
  GET  /api/exercises      - the full exercise database (JSON)
  POST /api/demo           - run a self-contained synthetic demo
  POST /api/analyze        - upload student (+ pro references) -> diagnosis + plan

Run locally:  uvicorn app:app --reload --port 8000
"""
import base64
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager

import numpy as np
import matplotlib
matplotlib.use("Agg")
# Use the object-oriented Figure API rather than pyplot: FastAPI runs sync
# endpoints in a thread pool, and pyplot keeps global per-process state, so two
# simultaneous requests could draw into each other's figure.
from matplotlib.figure import Figure

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from analysis_core import extract, build_corridor, UnreadableAudio
import classify as CL
from exercises import recommend, format_plan, _DB

log = logging.getLogger("tutor")

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff", ".aif")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024     # per file
MAX_PROS = 8                             # each reference costs one DTW pass
CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    """Warm the heavy audio libraries at boot in the background, so the first
    real request doesn't pay the one-time import/JIT cost."""
    import threading

    def run():
        try:
            from perf_synth import render, SR as SYNTH_SR
            a = extract(render(bpm=66, seed=1), sr=SYNTH_SR)
            b = extract(render(bpm=66, seed=2), sr=SYNTH_SR)
            cor = build_corridor([a, b])
            CL.classify(a, cor, [a, b])
        except Exception:
            log.exception("warm-up failed")

    threading.Thread(target=run, daemon=True).start()
    yield


app = FastAPI(title="AI Music Practice Tutor API", version="1.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------- helpers ------------------------------------
def _save_tmp(upload: UploadFile) -> str:
    """Stream an upload to disk with a hard size cap.

    The whole file used to be read into memory at once, so a single large POST
    could take down a 512 MB instance. CORS is open, so this endpoint has to
    assume the caller is not the front end.
    """
    suffix = os.path.splitext(upload.filename or "")[1].lower() or ".wav"
    if suffix not in AUDIO_EXT:
        raise HTTPException(400, "Unsupported audio type: %s" % suffix)
    fd, path = tempfile.mkstemp(suffix=suffix)
    written = 0
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = upload.file.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, "Each recording must be under %d MB."
                             % (MAX_UPLOAD_BYTES // (1024 * 1024)))
                f.write(chunk)
    except BaseException:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    if written == 0:
        os.remove(path)
        raise HTTPException(400, "Empty upload: %s" % (upload.filename or "file"))
    return path


def _plot_b64(corridor, alignment):
    n = corridor["n"]
    x = 100.0 * np.arange(n) / n
    panels = [("Intonation (cents)", "cents", "cents_mean", "cents_std"),
              ("Dynamics (dB)", "db", "db_mean", "db_std"),
              ("Tone: brightness (semitones)", "centroid", "centroid_mean", "centroid_std"),
              ("Timbre: breathiness (dB)", "flatness", "flatness_mean", "flatness_std")]
    fig = Figure(figsize=(10, 8))
    ax = fig.subplots(4, 1, sharex=True)
    for k, (title, key, mk, sk) in enumerate(panels):
        stu = alignment["curves"][key]              # already warped once, upstream
        dev = CL._rmed(stu - corridor[mk])
        thr = np.maximum(CL.K * corridor[sk], CL.FLOORS[key])
        ax[k].fill_between(x, -thr, thr, color="#9fd3a3", alpha=0.45,
                           label="pro corridor" if k == 0 else None)
        ax[k].plot(x, dev, color="#1f4e79", lw=1.2, label="you" if k == 0 else None)
        bad = (np.abs(dev) > thr) & corridor["valid"] & ~np.isnan(dev)
        ax[k].plot(x[bad], dev[bad], ".", color="#d64545", ms=4,
                   label="flagged" if k == 0 else None)
        ax[k].axhline(0, color="gray", lw=0.6)
        ax[k].set_ylabel(title, fontsize=9)
    ax[0].set_title("You vs the professional corridor", fontsize=12)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[-1].set_xlabel("position through the piece (%)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _payload(student, corridor, pros, name, instrument):
    # One alignment, reused by the classifier and the plot.
    alignment = CL.align(student, corridor)
    findings, summary = CL.classify(student, corridor, pros, alignment=alignment)
    plan = recommend(findings, summary, instrument)
    notes = []
    if summary.get("truncated"):
        notes.append("Only the first 45 seconds of each recording were analysed.")
    if summary.get("n_pros", 0) < 3:
        notes.append("With only %d reference recordings the professional corridor is "
                     "wide and approximate - 3 or more gives a much sharper diagnosis."
                     % summary.get("n_pros", 0))
    return {
        "name": os.path.basename(name),
        "instrument": instrument,
        "tempo_ratio": summary["tempo_ratio"],
        "vibrato_msgs": summary["vibrato_msgs"],
        "notes": notes,
        "findings": [
            {"category": f["category"], "label": f["label"], "value": f["value"],
             "t0": None if f["t0"] != f["t0"] else round(f["t0"], 1),
             "t1": None if f["t1"] != f["t1"] else round(f["t1"], 1),
             "pos": [round(f["pos_pct"][0]), round(f["pos_pct"][1])]}
            for f in findings],
        "plan": plan,
        "plan_text": format_plan(plan, instrument),
        "plot": _plot_b64(corridor, alignment),
    }


# ----------------------------- routes -------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "exercises": len(_DB["exercises"])}


@app.get("/api/exercises")
def exercises():
    return _DB


@app.post("/api/demo")
def demo(instrument: str = Form(None)):
    from perf_synth import render, PHRASE, SR as SYNTH_SR
    N = len(PHRASE)
    rng = np.random.default_rng(0)
    pros = [extract(render(bpm=float(rng.uniform(63, 69)), seed=10 + i), sr=SYNTH_SR)
            for i in range(5)]
    detune = list(rng.normal(0, 4, N)); detune[4] = -42
    breath = [0.0] * N
    for i in (0, 1, 2):
        breath[i] = 0.12
    arch = np.sin(np.linspace(0.15, np.pi - 0.15, N)); dyn = 0.5 + 0.5 * arch
    dyn[4] *= 0.5
    stu = extract(render(bpm=78, detune=detune, breath=breath, dyn=dyn, seed=99),
                  sr=SYNTH_SR)
    return _payload(stu, build_corridor(pros), pros, "demo_student.wav", instrument or None)


# Deliberately a sync `def`: FastAPI runs it in a worker thread. As `async def`
# the CPU-bound librosa work ran on the event loop and blocked every other
# request - including /api/health, which is what the platform pings to decide
# whether the service is alive.
@app.post("/api/analyze")
def analyze(student: UploadFile = File(...),
            pros: list[UploadFile] = File(...),
            instrument: str = Form(None)):
    if len(pros) < 2:
        raise HTTPException(400, "Please provide at least 2 professional recordings.")
    if len(pros) > MAX_PROS:
        raise HTTPException(400, "Please provide at most %d professional recordings."
                                 % MAX_PROS)
    paths = []
    try:
        for p in pros:
            paths.append(_save_tmp(p))
        pro_paths = list(paths)
        stu_path = _save_tmp(student)
        paths.append(stu_path)
        pro_feats = [extract(p) for p in pro_paths]
        corridor = build_corridor(pro_feats)
        stu_feat = extract(stu_path)
        return _payload(stu_feat, corridor, pro_feats, student.filename, instrument or None)
    except HTTPException:
        raise
    except UnreadableAudio as e:
        raise HTTPException(
            400, "One of those files could not be read as audio: %s. Try "
                 "exporting it as WAV or MP3." % e)
    except Exception as e:
        log.exception("analysis failed")
        raise HTTPException(500, "Could not analyse those recordings: %s" % e)
    finally:
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass

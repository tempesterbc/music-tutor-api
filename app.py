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
import gc
import io
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from analysis_core import extract, build_corridor, warp_map, warp_curve
import classify as CL
from exercises import recommend, format_plan, _DB

app = FastAPI(title="AI Music Practice Tutor API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff", ".aif")


# ----------------------------- helpers ------------------------------------
def _save_tmp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1].lower() or ".wav"
    if suffix not in AUDIO_EXT:
        raise HTTPException(400, "Unsupported audio type: %s" % suffix)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


def _plot_b64(student, corridor, name):
    n = corridor["n"]
    m = warp_map(corridor["ref_chroma"], student["chroma"])
    x = 100.0 * np.arange(n) / n
    panels = [("Intonation (cents)", "cents", "cents_mean", "cents_std", 20.0),
              ("Dynamics (dB)", "db", "db_mean", "db_std", 4.0),
              ("Tone: brightness (semitones)", "centroid", "centroid_mean", "centroid_std", 1.3),
              ("Timbre: breathiness (dB)", "flatness", "flatness_mean", "flatness_std", 3.0)]
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for k, (title, key, mk, sk, floor) in enumerate(panels):
        stu = warp_curve(student[key], m, n)
        dev = CL._rmed(stu - corridor[mk])
        thr = np.maximum(CL.K * corridor[sk], floor)
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
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=95)
    plt.close(fig)
    gc.collect()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _payload(student, corridor, pros, name, instrument):
    findings, summary = CL.classify(student, corridor, pros)
    plan = recommend(findings, summary, instrument)
    return {
        "name": os.path.basename(name),
        "instrument": instrument,
        "tempo_ratio": summary["tempo_ratio"],
        "vibrato_msgs": summary["vibrato_msgs"],
        "findings": [
            {"category": f["category"], "label": f["label"], "value": f["value"],
             "t0": None if f["t0"] != f["t0"] else round(f["t0"], 1),
             "t1": None if f["t1"] != f["t1"] else round(f["t1"], 1),
             "pos": [round(f["pos_pct"][0]), round(f["pos_pct"][1])]}
            for f in findings],
        "plan": plan,
        "plan_text": format_plan(plan, instrument),
        "plot": _plot_b64(student, corridor, name),
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
    from perf_synth import render, PHRASE
    N = len(PHRASE)
    rng = np.random.default_rng(0)
    pros = [extract(render(bpm=float(rng.uniform(63, 69)), seed=10 + i)) for i in range(5)]
    detune = list(rng.normal(0, 4, N)); detune[4] = -42
    breath = [0.0] * N
    for i in (0, 1, 2):
        breath[i] = 0.12
    arch = np.sin(np.linspace(0.15, np.pi - 0.15, N)); dyn = 0.5 + 0.5 * arch
    dyn[4] *= 0.5
    stu = extract(render(bpm=78, detune=detune, breath=breath, dyn=dyn, seed=99))
    return _payload(stu, build_corridor(pros), pros, "demo_student.wav", instrument or None)


@app.post("/api/analyze")
async def analyze(student: UploadFile = File(...),
                  pros: list[UploadFile] = File(...),
                  instrument: str = Form(None)):
    if len(pros) < 2:
        raise HTTPException(400, "Please provide at least 2 professional recordings.")
    paths = []
    try:
        pro_paths = [_save_tmp(p) for p in pros]
        stu_path = _save_tmp(student)
        paths = pro_paths + [stu_path]
        pro_feats = [extract(p) for p in pro_paths]
        corridor = build_corridor(pro_feats)
        stu_feat = extract(stu_path)
        return _payload(stu_feat, corridor, pro_feats, student.filename, instrument or None)
    finally:
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass


@app.on_event("startup")
def _warmup():
    """Compile/load the heavy audio libraries in the background at boot, so the
    first real request doesn't pay the one-time warm-up cost."""
    import threading

    def run():
        try:
            from perf_synth import render
            a = extract(render(bpm=66, seed=1))
            b = extract(render(bpm=66, seed=2))
            cor = build_corridor([a, b])
            CL.classify(a, cor, [a, b])
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()

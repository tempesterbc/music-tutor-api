"""
exercises.py - recommend targeted practice exercises for detected problems, using
the user's curated Musical Exercise Database (exercise_database.json).

The detector reports problems in these categories:
    intonation, tempo, rhythm, dynamics, tone, timbre, evenness, vibrato
Each is mapped to specific exercises drawn from the database. Woodwind-specific
"deep dive" exercises (W1-W22) are preferred when an instrument is given.
"""
import json
import os

_DB = json.load(open(os.path.join(os.path.dirname(__file__), "exercise_database.json")))
_BY_NAME = {e["exercise"]: e for e in _DB["exercises"]}
_BY_WID = {w["id"]: w for w in _DB["woodwind"]}

# ---- map (category, direction) -> ordered list of exercise refs -------------
# refs are either a main-DB exercise name or a woodwind id like "W3".
PROBLEM_MAP = {
    ("intonation", ""): ["Long Tones with Tuner/Drone", "Slow Scales with Drone",
                          "Singing Over a Drone"],
    ("tempo", ""):      ["Rudiments with Metronome (all 40)",
                         "Clapping/Tapping Rhythm Along to Music"],
    ("rhythm", ""):     ["Rudiments with Metronome (all 40)",
                         "Clapping/Tapping Rhythm Along to Music"],
    ("dynamics", ""):   ["Messa di Voce", "Crescendo–Decrescendo Long Tones",
                         "Dynamic Terracing on Scales"],
    ("tone", "dull"):   ["Long Tones", "Overtone Series Studies",
                         "Flute Harmonics / Embouchure Flexibility"],
    ("tone", "bright"): ["Overtone Series Studies", "Long Tones",
                         "Lip Slurs / Lip Flexibilities"],
    ("timbre", "breathy"): ["Diaphragmatic (Belly) Breathing",
                            "Straw Breathing / Pursed-Lip Resistance", "Long Tones"],
    ("timbre", "pressed"): ["Yawn-Sigh Technique", "Long Tones",
                            "Overtone Series Studies"],
    ("timbre", ""):     ["Long Tones", "Overtone Series Studies"],
    ("evenness", ""):   ["Scales (Major/Minor/Chromatic)",
                         "Hanon Exercises (The Virtuoso Pianist)",
                         "Thirds and Interval Studies"],
    ("vibrato", ""):    ["Vibrato Isolation Drills (Knocking Motion)", "Messa di Voce"],
}

# woodwind deep-dive exercises to PREFER (prepend) per category when instrument is a woodwind
WOODWIND_BOOST = {
    "intonation": ["W3", "W1"],
    "dynamics":   ["W3"],
    "tone":       ["W1", "W15"],
    "timbre":     ["W2"],
    "evenness":   ["W4"],
    "vibrato":    ["W9", "W22"],
}
WOODWINDS = ("flute", "clarinet", "sax", "saxophone", "oboe", "bassoon", "woodwind")
# which woodwind vibrato exercise fits which instrument
_VIB_BY_INST = {"flute": "W9", "oboe": "W22", "bassoon": "W22"}

PRIORITY = {"intonation": 5, "tempo": 5, "rhythm": 5, "tone": 4, "timbre": 4,
            "evenness": 3, "dynamics": 3, "vibrato": 2}


def _direction(category, label):
    lab = label.lower()
    if category == "tone":
        return "bright" if ("bright" in lab or "harsh" in lab) else "dull"
    if category == "timbre":
        if "breath" in lab or "airy" in lab:
            return "breathy"
        if "pressed" in lab or "edgy" in lab:
            return "pressed"
    return ""


def _resolve(ref):
    """Turn an exercise ref into a normalized record."""
    if ref in _BY_WID:
        w = _BY_WID[ref]
        return {"name": "%s (%s)" % (w["exercise"], w["id"]), "category": "Woodwind",
                "instrument": w["instrument"], "develops": w["primary"],
                "also": w["secondary"], "how": w["how"], "difficulty": w["difficulty"]}
    e = _BY_NAME.get(ref)
    if not e:
        return None
    return {"name": e["exercise"], "category": e["category"],
            "instrument": e["instrument"], "develops": e["primary"],
            "also": e["secondary"], "how": e["how"], "difficulty": e["difficulty"]}


def _exercises_for(category, direction, instrument, limit=2):
    refs = list(PROBLEM_MAP.get((category, direction))
                or PROBLEM_MAP.get((category, "")) or [])
    inst = (instrument or "").lower()
    if any(w in inst for w in WOODWINDS):
        boost = list(WOODWIND_BOOST.get(category, []))
        if category == "vibrato":
            wid = next((v for k, v in _VIB_BY_INST.items() if k in inst), None)
            boost = [wid] if wid else []
        refs = boost + [r for r in refs if r not in boost]
    seen, recs = set(), []
    for r in refs:
        if r in seen:
            continue
        seen.add(r)
        rec = _resolve(r)
        if rec:
            recs.append(rec)
        if len(recs) >= limit:
            break
    return recs


def recommend(findings, summary, instrument=None):
    """Return a prioritised practice plan drawn from the user's exercise DB."""
    agg = {}
    tr = summary.get("tempo_ratio", 1.0)
    if abs(tr - 1) > 0.05:
        agg[("tempo", "")] = {"problem": "playing %d%% %s than pros"
                              % (round(abs(tr - 1) * 100), "slower" if tr > 1 else "faster"),
                              "count": 1, "severity": abs(tr - 1) * 100}
    for msg in summary.get("vibrato_msgs", []):
        agg[("vibrato", "")] = {"problem": msg, "count": 1, "severity": 1.0}
    for f in findings:
        cat = f["category"]
        d = _direction(cat, f["label"])
        e = agg.setdefault((cat, d), {"problem": "%s (%s)" % (f["label"], cat),
                                      "count": 0, "severity": 0.0})
        e["count"] += 1
        e["severity"] = max(e["severity"], f.get("severity", 1.0))

    plan = []
    for (cat, d), info in agg.items():
        recs = _exercises_for(cat, d, instrument)
        if not recs:
            continue
        plan.append({"category": cat, "direction": d, "problem": info["problem"],
                     "count": info["count"], "severity": info["severity"],
                     "exercises": recs})
    plan.sort(key=lambda p: (PRIORITY.get(p["category"], 1), p["severity"], p["count"]),
              reverse=True)
    return plan


def format_plan(plan, instrument=None):
    if not plan:
        return ("No targeted exercises needed - performance is inside the "
                "professional range. Maintain with daily Long Tones.")
    head = "PERSONALISED PRACTICE PLAN (from your Musical Exercise Database"
    head += (", %s)" % instrument) if instrument else ")"
    L = [head, ""]
    for i, p in enumerate(plan, 1):
        tag = (" - flagged %d place(s)" % p["count"]) if p["count"] > 1 else ""
        L.append("%d. Problem: %s%s" % (i, p["problem"], tag))
        for ex in p["exercises"]:
            L.append("   > %s  [%s | %s]" % (ex["name"], ex["category"], ex["difficulty"]))
            L.append("     Develops: %s" % ex["develops"])
            L.append("     How: %s" % ex["how"])
        L.append("")
    return "\n".join(L)

"""BleedAlert API.

Run:  uvicorn app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import base64
import io
import os
import sqlite3
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ct_utils import load_scan, study_key
from model import LABELS, Predictor
from simulate import simulate

DB_PATH = os.path.join(os.path.dirname(__file__), "bleedalert.db")
URGENT = 0.65      # at or above: urgent, jump the queue
UNSURE_LOW = 0.35  # between UNSURE_LOW and URGENT: model is unsure, send to a human

app = FastAPI(title="BleedAlert")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = Predictor()


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY, filename TEXT, uploaded_at REAL,
            p_any REAL, subtype TEXT, status TEXT, summary TEXT,
            scan_png TEXT, heatmap_png TEXT, reviewed INTEGER DEFAULT 0,
            study TEXT, slices INTEGER DEFAULT 1)"""
    )
    return conn


def triage(p_any):
    if p_any >= URGENT:
        return "URGENT"
    if p_any >= UNSURE_LOW:
        return "NEEDS HUMAN REVIEW"
    return "ROUTINE"


def summarize(p_any, subtype, status):
    """Gemini free tier if GEMINI_API_KEY is set, otherwise a template."""
    fallback = (
        f"Triage {status}. Model estimates {p_any:.0%} probability of intracranial hemorrhage"
        + (f", most consistent with {subtype}." if status != "ROUTINE" else ".")
        + " This is triage support only. A physician must review the images."
    )
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return fallback
    try:
        from google import genai

        client = genai.Client(api_key=key)
        prompt = (
            "Write two short sentences for an on call neurologist. Do not diagnose. "
            f"Triage status: {status}. Bleed probability: {p_any:.0%}. Most likely subtype: {subtype}. "
            "End by saying a physician must review the images."
        )
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return resp.text.strip()
    except Exception:
        return fallback


@app.get("/health")
def health():
    return {"ok": True, "demo_mode": predictor.demo}


@app.get("/simulate")
def run_simulation(scans_per_day: int = 120, read_minutes: float = 8.0):
    """Average wait for bleed patients, first come first served vs BleedAlert triage."""
    return simulate(scans_per_day=scans_per_day, read_minutes=read_minutes)


@app.post("/scans")
async def upload_scan(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        img = load_scan(raw, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not read scan: {e}")

    scores, heatmap_b64 = predictor.predict(img)
    p_any = scores["any"]
    subtype = max(LABELS[1:], key=lambda k: scores[k])
    status = triage(p_any)
    summary = summarize(p_any, subtype, status)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    scan_b64 = base64.b64encode(buf.getvalue()).decode()

    study = study_key(raw, file.filename)
    with db() as conn:
        row = conn.execute("SELECT * FROM scans WHERE study = ? AND reviewed = 0", (study,)).fetchone()
        if row is None:
            scan_id = uuid.uuid4().hex[:8]
            conn.execute(
                "INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,0,?,1)",
                (scan_id, file.filename, time.time(), p_any, subtype, status, summary,
                 scan_b64, heatmap_b64, study),
            )
        else:
            # Another slice of a study already in the queue: the study's score is its most
            # suspicious slice, and that slice's image and heatmap are the ones shown.
            scan_id = row["id"]
            conn.execute("UPDATE scans SET slices = slices + 1 WHERE id = ?", (scan_id,))
            if p_any > row["p_any"]:
                conn.execute(
                    "UPDATE scans SET filename=?, p_any=?, subtype=?, status=?, summary=?, "
                    "scan_png=?, heatmap_png=? WHERE id=?",
                    (file.filename, p_any, subtype, status, summary, scan_b64, heatmap_b64, scan_id),
                )
    return {"id": scan_id, "study": study, "status": status, "p_any": p_any,
            "subtype": subtype, "scores": scores, "summary": summary}


@app.get("/worklist")
def worklist():
    """Unreviewed scans, most urgent first, then oldest first."""
    order = {"URGENT": 0, "NEEDS HUMAN REVIEW": 1, "ROUTINE": 2}
    with db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM scans WHERE reviewed = 0")]
    rows.sort(key=lambda r: (order[r["status"]], r["uploaded_at"]))
    return rows


@app.post("/scans/{scan_id}/review")
def mark_reviewed(scan_id: str):
    with db() as conn:
        conn.execute("UPDATE scans SET reviewed = 1 WHERE id = ?", (scan_id,))
    return {"ok": True}


@app.delete("/worklist")
def reset():
    with db() as conn:
        conn.execute("DELETE FROM scans")
    return {"ok": True}


from fastapi.responses import FileResponse


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

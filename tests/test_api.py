"""End-to-end API tests using the synthetic demo scans (no patient data)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_client(tmp_path):
    os.environ["BLEEDALERT_DB"] = str(tmp_path / "test.db")
    import importlib

    import app as app_module

    importlib.reload(app_module)
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


def test_health_reports_trained_model(tmp_path):
    c = make_client(tmp_path)
    h = c.get("/health").json()
    assert h["ok"] is True
    assert h["demo_mode"] is False, "bleedalert_weights.npz should be loaded"


def test_demo_scans_are_scored_and_sorted(tmp_path):
    c = make_client(tmp_path)
    r = c.post("/demo/load").json()
    assert r["loaded"] == 4
    for res in r["results"]:
        assert 0.0 <= res["p_any"] <= 1.0
        assert res["status"] in {"URGENT", "NEEDS HUMAN REVIEW", "ROUTINE"}
    rows = c.get("/worklist").json()
    order = {"URGENT": 0, "NEEDS HUMAN REVIEW": 1, "ROUTINE": 2}
    keys = [order[r["status"]] for r in rows]
    assert keys == sorted(keys), "worklist must put the most urgent cases first"
    assert all(r["heatmap_png"] for r in rows), "every case should carry a Grad-CAM heatmap"


def test_review_removes_case_from_worklist(tmp_path):
    c = make_client(tmp_path)
    c.post("/demo/load")
    first = c.get("/worklist").json()[0]
    c.post(f"/scans/{first['id']}/review")
    assert first["id"] not in [r["id"] for r in c.get("/worklist").json()]


def test_bad_upload_is_rejected(tmp_path):
    c = make_client(tmp_path)
    r = c.post("/scans", files={"file": ("junk.dcm", b"not a dicom", "application/dicom")})
    assert r.status_code == 400


def test_simulation_shows_triage_helps(tmp_path):
    c = make_client(tmp_path)
    s = c.get("/simulate?scans_per_day=120").json()
    assert s["triage_avg_wait_min"] < s["fifo_avg_wait_min"]

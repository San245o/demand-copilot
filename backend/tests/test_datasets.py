from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app.adapters.uploaded import UploadedAdapter, save_uploaded_dataset
from app.core.config import settings
from app.main import app
from app.services.profiler import profile_dataset


def test_heuristic_profiler_relevant(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "sku": ["A", "A", "A"],
            "sales": [10, 12, 14],
            "promo": [0, 1, 0],
        }
    )
    profile = profile_dataset(df, "sales.csv")
    assert profile.relevant is True
    assert profile.date_column == "date"
    assert profile.target_column == "sales"
    assert "promo" in profile.signal_columns


def test_heuristic_profiler_flags_irrelevant(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)
    df = pd.DataFrame({"name": ["a", "b"], "category": ["x", "y"]})
    profile = profile_dataset(df, "notes.csv")
    assert profile.relevant is False
    assert "date" in profile.relevance_reason.lower()


def test_uploaded_adapter_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "store": ["A", "A", "A"],
            "sales": [10, 5, 12],
            "promo": [1, 1, 0],
        }
    )
    profile = profile_dataset(df, "roundtrip.csv")
    save_uploaded_dataset(df, profile, root=tmp_path)
    adapter = UploadedAdapter(profile.id, root=tmp_path)
    assert adapter.list_entities() == ["A"]
    series = adapter.load("A")
    assert series.entity_id == "A"
    assert [p.target for p in series.points] == [15.0, 12.0]


def test_upload_activate_entities_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)
    client = TestClient(app)
    csv = "date,store,sales,promo\n2024-01-01,A,10,0\n2024-01-02,A,12,1\n"
    uploaded_id = None
    try:
        r = client.post(
            "/datasets/upload",
            files=[("files", ("endpoint.csv", csv, "text/csv"))],
        )
        assert r.status_code == 200
        uploaded_id = r.json()["profiles"][0]["id"]
        assert r.json()["profiles"][0]["relevant"] is True

        r = client.post(f"/datasets/{uploaded_id}/activate")
        assert r.status_code == 200
        assert r.json()["active_id"] == uploaded_id

        r = client.get("/entities")
        assert r.status_code == 200
        assert r.json()["adapter"] == uploaded_id
        assert r.json()["entities"] == ["A"]
    finally:
        if uploaded_id:
            client.delete(f"/datasets/{uploaded_id}")

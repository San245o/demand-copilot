from __future__ import annotations

import json
import shutil

from app.adapters.base import DatasetAdapter
from app.adapters.rossmann import RossmannAdapter
from app.adapters.synthetic import SyntheticAdapter
from app.adapters.uploaded import UPLOAD_ROOT, UploadedAdapter, load_uploaded_profile
from app.core.schema import DatasetProfile, DatasetRegion


ACTIVE_FILE = UPLOAD_ROOT / "_active.json"


def list_datasets() -> list[DatasetProfile]:
    profiles: list[DatasetProfile] = []
    rossmann = RossmannAdapter()
    if rossmann.available:
        profiles.append(_rossmann_profile())
    profiles.append(_synthetic_profile())
    profiles.extend(_uploaded_profiles())
    return profiles


def get_active_profile() -> DatasetProfile:
    active = _read_active_id()
    profiles = list_datasets()
    if active:
        for profile in profiles:
            if profile.id == active:
                return profile
    default = _default_dataset_id()
    for profile in profiles:
        if profile.id == default:
            return profile
    return profiles[0]


def set_active_dataset(dataset_id: str) -> DatasetProfile:
    for profile in list_datasets():
        if profile.id == dataset_id:
            UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
            with ACTIVE_FILE.open("w", encoding="utf-8") as f:
                json.dump({"id": dataset_id}, f)
            return profile
    raise ValueError(f"Unknown dataset '{dataset_id}'")


def delete_dataset(dataset_id: str) -> None:
    if not dataset_id.startswith("upload-"):
        raise ValueError("Only uploaded datasets can be deleted")
    target = UPLOAD_ROOT / dataset_id
    if not target.exists():
        raise ValueError(f"Unknown dataset '{dataset_id}'")
    shutil.rmtree(target)
    if _read_active_id() == dataset_id:
        if ACTIVE_FILE.exists():
            ACTIVE_FILE.unlink()


def get_adapter(name: str | None = None) -> DatasetAdapter:
    """Return a dataset adapter.

    Default selection prefers real Rossmann data when present, else falls back to
    the synthetic adapter so the pipeline always runs. Pass an explicit name to force.
    """
    if name is None:
        name = get_active_profile().id
    if name and name.startswith("upload-"):
        return UploadedAdapter(name)
    if name == "synthetic":
        return SyntheticAdapter()
    if name == "rossmann":
        return RossmannAdapter()

    rossmann = RossmannAdapter()
    if rossmann.available:
        return rossmann
    return SyntheticAdapter()


def active_adapter_name() -> str:
    return get_adapter().name


def _default_dataset_id() -> str:
    return "rossmann" if RossmannAdapter().available else "synthetic"


def _read_active_id() -> str | None:
    try:
        with ACTIVE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        dataset_id = data.get("id")
        return dataset_id if isinstance(dataset_id, str) else None
    except Exception:
        return None


def _uploaded_profiles() -> list[DatasetProfile]:
    if not UPLOAD_ROOT.exists():
        return []
    profiles: list[DatasetProfile] = []
    for path in sorted(UPLOAD_ROOT.iterdir()):
        if path.is_dir() and (path / "profile.json").exists():
            try:
                profiles.append(load_uploaded_profile(path.name))
            except Exception:
                continue
    return profiles


def _rossmann_profile() -> DatasetProfile:
    return DatasetProfile(
        id="rossmann",
        name="Rossmann Store Sales",
        source_file="train.csv",
        relevant=True,
        relevance_reason="Bundled retail store sales time series.",
        description="Daily Rossmann store sales with promo and holiday signals.",
        date_column="Date",
        target_column="Sales",
        target_name="sales",
        unit="euros",
        entity_columns=["Store"],
        signal_columns=["Promo", "SchoolHoliday", "StateHoliday"],
        freq="D",
        region=DatasetRegion(country_code="DE", country_name="Germany", lat=51.0, lon=9.0),
        row_count=0,
        suggested_questions=[
            "Forecast demand for store_1 over the next 7 days and explain the drivers.",
            "Which stores can I forecast?",
            "How do promotions affect demand? Check the playbooks.",
        ],
    )


def _synthetic_profile() -> DatasetProfile:
    return DatasetProfile(
        id="synthetic",
        name="Synthetic Retail Demand",
        source_file="generated",
        relevant=True,
        relevance_reason="Deterministic fallback demand time series.",
        description="Synthetic daily store demand with weekly seasonality, trend, promo spikes, and holiday dips.",
        date_column="date",
        target_column="target",
        target_name="sales",
        unit="units",
        entity_columns=["store"],
        signal_columns=["promo", "holiday"],
        freq="D",
        region=None,
        row_count=365 * 5,
        suggested_questions=[
            "Forecast sales for store_1 over the next 7 days.",
            "What signals are available in this dataset?",
            "What recent trend should I watch for store_3?",
        ],
    )

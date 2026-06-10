from __future__ import annotations

from app.adapters.base import DatasetAdapter
from app.adapters.rossmann import RossmannAdapter
from app.adapters.synthetic import SyntheticAdapter


def get_adapter(name: str | None = None) -> DatasetAdapter:
    """Return a dataset adapter.

    Default selection prefers real Rossmann data when present, else falls back to
    the synthetic adapter so the pipeline always runs. Pass an explicit name to force.
    """
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

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.core.schema import TimeSeries


@runtime_checkable
class DatasetAdapter(Protocol):
    """Maps a specific dataset into the canonical schema.

    This is the ONLY dataset-aware code in the system. Agents and tools never see
    raw dataset columns — only the canonical TimeSeries an adapter produces. Adding
    a new dataset (Favorita, M5, a customer CSV, a live warehouse) = one new adapter,
    nothing else changes.
    """

    name: str

    def list_entities(self) -> list[str]:
        """Return the forecastable entity ids this adapter can serve."""
        ...

    def load(self, entity_id: str, lookback: int | None = None) -> TimeSeries:
        """Return the canonical series for one entity (most recent `lookback` points)."""
        ...

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.core.schema import DataPoint, DatasetProfile, TimeSeries


UPLOAD_ROOT = Path(__file__).resolve().parents[2] / "data" / "uploads"


class UploadedAdapter:
    def __init__(self, dataset_id: str, root: Path = UPLOAD_ROOT) -> None:
        self.dataset_id = dataset_id
        self.root = root
        self.dataset_dir = root / dataset_id
        self.name = dataset_id
        self.profile = self._load_profile()

    def list_entities(self) -> list[str]:
        df = self._load_df()
        entity = self._entity_series(df)
        return sorted(entity.dropna().unique().tolist())[:50]

    def load(self, entity_id: str, lookback: int | None = None) -> TimeSeries:
        if not self.profile.date_column or not self.profile.target_column:
            raise ValueError("Uploaded dataset is missing date or target mapping")
        df = self._load_df()
        df["__date"] = pd.to_datetime(df[self.profile.date_column], errors="coerce")
        df["__target"] = pd.to_numeric(df[self.profile.target_column], errors="coerce")
        df["__entity"] = self._entity_series(df)
        df = df[(df["__entity"] == entity_id) & df["__date"].notna() & df["__target"].notna()]
        if df.empty:
            raise ValueError(f"Unknown entity {entity_id}")

        signal_cols = [c for c in self.profile.signal_columns if c in df.columns]
        for col in signal_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        agg = {"__target": "sum", **{c: "mean" for c in signal_cols}}
        grouped = df.groupby("__date", as_index=False).agg(agg).sort_values("__date")
        if lookback:
            grouped = grouped.tail(lookback)

        points: list[DataPoint] = []
        for _, row in grouped.iterrows():
            signals = {
                c: float(row[c])
                for c in signal_cols
                if pd.notna(row.get(c))
            }
            points.append(
                DataPoint(
                    date=pd.Timestamp(row["__date"]).date(),
                    target=float(row["__target"]),
                    signals=signals,
                )
            )
        return TimeSeries(
            entity_id=entity_id,
            entity_kind="_".join(self.profile.entity_columns) or "entity",
            freq=self.profile.freq,
            target_name=self.profile.target_name,
            unit=self.profile.unit,
            points=points,
        )

    def _load_profile(self) -> DatasetProfile:
        with (self.dataset_dir / "profile.json").open("r", encoding="utf-8") as f:
            return DatasetProfile.model_validate(json.load(f))

    def _load_df(self) -> pd.DataFrame:
        return pd.read_csv(self.dataset_dir / "data.csv")

    def _entity_series(self, df: pd.DataFrame) -> pd.Series:
        cols = [c for c in self.profile.entity_columns if c in df.columns]
        if not cols:
            return pd.Series("all", index=df.index)
        return df[cols].astype(str).agg(" | ".join, axis=1)


def save_uploaded_dataset(df: pd.DataFrame, profile: DatasetProfile, root: Path = UPLOAD_ROOT) -> DatasetProfile:
    dataset_dir = root / profile.id
    dataset_dir.mkdir(parents=True, exist_ok=False)
    df.to_csv(dataset_dir / "data.csv", index=False)
    with (dataset_dir / "profile.json").open("w", encoding="utf-8") as f:
        json.dump(profile.model_dump(mode="json"), f, indent=2)
    return profile


def load_uploaded_profile(dataset_id: str, root: Path = UPLOAD_ROOT) -> DatasetProfile:
    with (root / dataset_id / "profile.json").open("r", encoding="utf-8") as f:
        return DatasetProfile.model_validate(json.load(f))

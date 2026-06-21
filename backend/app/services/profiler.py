from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.schema import DatasetProfile, DatasetRegion


class _ProfileDraft(BaseModel):
    relevant: bool
    relevance_reason: str
    description: str
    date_column: str | None = None
    target_column: str | None = None
    target_name: str = "demand"
    unit: str | None = None
    entity_columns: list[str] = Field(default_factory=list)
    signal_columns: list[str] = Field(default_factory=list)
    freq: str = "D"
    region: DatasetRegion | None = None
    suggested_questions: list[str] = Field(default_factory=list)


def summarize_dataframe(df: pd.DataFrame) -> str:
    """Return the compact table profile shown to the LLM."""
    sample = df.head(8).where(pd.notna(df.head(8)), None).to_dict(orient="records")
    lines = [f"Rows: {len(df)}", "Columns:"]
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        null_pct = 0.0 if len(s) == 0 else round(float(s.isna().mean() * 100), 1)
        stats = [f"dtype={s.dtype}", f"nunique={s.nunique(dropna=True)}", f"null_pct={null_pct}"]
        if pd.api.types.is_numeric_dtype(s) and not non_null.empty:
            stats.extend([f"min={non_null.min()}", f"max={non_null.max()}"])
        lines.append(f"- {col}: " + ", ".join(stats))
    lines.append("Sample rows:")
    lines.append(str(sample))
    return "\n".join(lines)


def profile_dataset(df: pd.DataFrame, filename: str) -> DatasetProfile:
    """Profile a user upload, using Gemini when available and heuristics otherwise."""
    df = _clean_columns(df)
    base = _heuristic_profile(df, filename)
    draft: _ProfileDraft | None = None
    if settings.has_llm:
        try:
            draft = _llm_profile(df, filename)
        except Exception:
            draft = None
    if draft is None:
        return base
    return _profile_from_draft(df, filename, draft, base)


def _llm_profile(df: pd.DataFrame, filename: str) -> _ProfileDraft:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=settings.google_api_key,
        temperature=0,
        max_retries=1,
        timeout=45,
    ).with_structured_output(_ProfileDraft)
    prompt = f"""
You profile uploaded tabular data for a demand forecasting chat copilot.

Given only headers, dtypes, compact stats, and sample rows, decide whether the file
is relevant for demand forecasting or demand analysis. Warn, do not block: relevant
should be false only when there is no plausible date/time plus numeric demand-like
measure. Identify:
- date_column: parseable time column
- target_column: numeric demand/sales/orders/revenue/quantity column to forecast
- entity_columns: low-cardinality columns defining forecastable entities, such as store, sku, product, region, customer segment
- signal_columns: observed drivers/covariates such as promo, holiday, price, stock, weather, event flags
- freq: one of D, W, M when inferable
- region: infer country and approximate lat/lon from country/city/currency/language clues; null when unguessable
- description: 1-2 sentences
- suggested_questions: exactly 3 useful chat questions for this dataset

Only return column names that exactly exist in the file.

Filename: {filename}
{summarize_dataframe(df)}
"""
    return llm.invoke(prompt)


def _profile_from_draft(
    df: pd.DataFrame, filename: str, draft: _ProfileDraft, base: DatasetProfile
) -> DatasetProfile:
    cols = set(df.columns)
    date_col = draft.date_column if draft.date_column in cols and _parse_dates(df[draft.date_column]).notna().any() else base.date_column
    target_col = draft.target_column if draft.target_column in cols and pd.api.types.is_numeric_dtype(pd.to_numeric(df[draft.target_column], errors="coerce")) else base.target_column
    entities = [c for c in draft.entity_columns if c in cols and c not in {date_col, target_col}]
    entities = _choose_entity_grain(df, date_col, target_col, entities)
    signals = [c for c in draft.signal_columns if c in cols and c not in {date_col, target_col} and c not in entities]
    dmin, dmax = _date_bounds(df, date_col)
    relevant = bool(date_col and target_col and draft.relevant)
    reason = draft.relevance_reason or base.relevance_reason
    if not date_col or not target_col:
        relevant = False
        reason = "Could not identify both a parseable date column and numeric target column."
    return DatasetProfile(
        id=_dataset_id(filename),
        name=Path(filename).stem.replace("_", " ").strip() or "uploaded dataset",
        source_file=filename,
        relevant=relevant,
        relevance_reason=reason,
        description=draft.description or base.description,
        date_column=date_col,
        target_column=target_col,
        target_name=draft.target_name or base.target_name,
        unit=draft.unit or base.unit,
        entity_columns=entities or base.entity_columns,
        signal_columns=signals or base.signal_columns,
        freq=_safe_freq(draft.freq or base.freq),
        region=draft.region or base.region,
        row_count=len(df),
        date_min=dmin,
        date_max=dmax,
        suggested_questions=_questions(draft.suggested_questions, target_col, entities),
    )


def _heuristic_profile(df: pd.DataFrame, filename: str) -> DatasetProfile:
    date_col = _best_date_column(df)
    target_col = _best_target_column(df, date_col)
    entity_cols = _entity_columns(df, date_col, target_col)
    entity_cols = _choose_entity_grain(df, date_col, target_col, entity_cols)
    signal_cols = _signal_columns(df, date_col, target_col, entity_cols)
    dmin, dmax = _date_bounds(df, date_col)
    relevant = bool(date_col and target_col)
    region = _infer_region(df)
    reason = (
        "Found a parseable date column and numeric target suitable for forecasting."
        if relevant
        else "Could not identify both a parseable date column and numeric target column."
    )
    target_name = _target_name(target_col)
    return DatasetProfile(
        id=_dataset_id(filename),
        name=Path(filename).stem.replace("_", " ").strip() or "uploaded dataset",
        source_file=filename,
        relevant=relevant,
        relevance_reason=reason,
        description=_description(filename, target_col, entity_cols, date_col),
        date_column=date_col,
        target_column=target_col,
        target_name=target_name,
        unit=_unit(target_col),
        entity_columns=entity_cols,
        signal_columns=signal_cols,
        freq=_infer_freq(df, date_col),
        region=region,
        row_count=len(df),
        date_min=dmin,
        date_max=dmax,
        suggested_questions=_questions([], target_col, entity_cols),
    )


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _dataset_id(filename: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-") or "dataset"
    return f"upload-{stem}-{uuid.uuid4().hex[:8]}"


def _best_date_column(df: pd.DataFrame) -> str | None:
    candidates = list(df.columns)
    candidates.sort(key=lambda c: 0 if re.search(r"date|time|day|week|month", c, re.I) else 1)
    best: tuple[float, str] | None = None
    for col in candidates:
        parsed = _parse_dates(df[col])
        ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
        if ratio >= 0.6 and (best is None or ratio > best[0]):
            best = (ratio, col)
    return best[1] if best else None


def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        name = str(s.name or "")
        if not re.search(r"date|time|day|week|month|year", name, re.I):
            return pd.Series(pd.NaT, index=s.index)
    return pd.to_datetime(s, errors="coerce")


def _best_target_column(df: pd.DataFrame, date_col: str | None) -> str | None:
    preferred = re.compile(r"sales|demand|orders?|units?|qty|quantity|revenue|amount|volume", re.I)
    numeric: list[tuple[int, float, str]] = []
    for col in df.columns:
        if col == date_col:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().mean() < 0.6:
            continue
        var = float(vals.var(skipna=True) or 0.0)
        if var <= 0:
            continue
        numeric.append((0 if preferred.search(col) else 1, -var, col))
    numeric.sort()
    return numeric[0][2] if numeric else None


def _entity_columns(df: pd.DataFrame, date_col: str | None, target_col: str | None) -> list[str]:
    out: list[str] = []
    preferred = re.compile(r"store|sku|product|item|entity|location|region|city|customer|segment|warehouse|channel", re.I)
    signalish = re.compile(r"promo|holiday|event|price|stock|weather|temp|discount|open|campaign", re.I)
    for col in df.columns:
        if col in {date_col, target_col}:
            continue
        if signalish.search(col):
            continue
        nunique = df[col].nunique(dropna=True)
        is_preferred = bool(preferred.search(col))
        if len(df) == 0 or (nunique <= 1 and not is_preferred):
            continue
        low_card = nunique <= min(50, max(2, len(df) // 2))
        stringy = pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        if low_card and (stringy or is_preferred):
            out.append(col)
        if len(out) >= 3:
            break
    return out


def _entity_series_lengths(df: pd.DataFrame, cols: list[str], date_col: str | None):
    """Median number of distinct dates per entity group (the real series length)."""
    present = [c for c in cols if c in df.columns]
    if not present or not date_col or date_col not in df.columns:
        return None
    return df.groupby(present)[date_col].nunique()


def _choose_entity_grain(
    df: pd.DataFrame, date_col: str | None, target_col: str | None,
    cands: list[str], min_points: int = 24,
) -> list[str]:
    """Pick an entity grain that actually has enough history to forecast.

    Combining every id-like column (e.g. product × category × store) often shatters
    the data into thousands of 1–2 row groups that cannot be forecast. If the full
    combination is too sparse, fall back to the most granular SINGLE column whose
    median series is long enough; if none qualifies, the column with the longest
    median series. This is what makes per-entity forecasts real instead of flat.
    """
    cands = [c for c in cands if c in df.columns and c not in {date_col, target_col}]
    if len(cands) <= 1 or not date_col or date_col not in df.columns:
        return cands
    full = _entity_series_lengths(df, cands, date_col)
    if full is not None and float(full.median()) >= min_points:
        return cands
    scored: list[tuple[str, int, float]] = []
    for c in cands:
        lengths = _entity_series_lengths(df, [c], date_col)
        if lengths is None or lengths.empty:
            continue
        scored.append((c, int(df[c].nunique(dropna=True)), float(lengths.median())))
    if not scored:
        return cands[:1]
    forecastable = [s for s in scored if s[2] >= min_points]
    if forecastable:
        best = max(forecastable, key=lambda s: s[1])  # most granular that's forecastable
    else:
        best = max(scored, key=lambda s: s[2])  # coarsest (longest series)
    return [best[0]]


def _signal_columns(
    df: pd.DataFrame, date_col: str | None, target_col: str | None, entity_cols: list[str]
) -> list[str]:
    out: list[str] = []
    preferred = re.compile(r"promo|holiday|event|price|stock|weather|temp|discount|open|campaign", re.I)
    skip = {date_col, target_col, *entity_cols}
    for col in df.columns:
        if col in skip:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        numericish = vals.notna().mean() >= 0.6
        binary = numericish and set(vals.dropna().unique()).issubset({0, 1, 0.0, 1.0})
        if binary or (preferred.search(col) and numericish):
            out.append(col)
    return out[:12]


def _date_bounds(df: pd.DataFrame, date_col: str | None):
    if not date_col:
        return None, None
    parsed = _parse_dates(df[date_col]).dropna()
    if parsed.empty:
        return None, None
    return parsed.min().date(), parsed.max().date()


def _infer_freq(df: pd.DataFrame, date_col: str | None) -> str:
    if not date_col:
        return "D"
    dates = _parse_dates(df[date_col]).dropna().sort_values().drop_duplicates()
    if len(dates) < 2:
        return "D"
    days = dates.diff().dt.days.dropna()
    if days.empty:
        return "D"
    median = float(days.median())
    if median >= 25:
        return "M"
    if median >= 6:
        return "W"
    return "D"


def _safe_freq(freq: str) -> str:
    return freq if freq in {"D", "W", "M"} else "D"


def _target_name(col: str | None) -> str:
    return (col or "demand").replace("_", " ").lower()


def _unit(col: str | None) -> str | None:
    if not col:
        return None
    low = col.lower()
    if any(x in low for x in ["sales", "revenue", "amount", "price"]):
        return "currency"
    if any(x in low for x in ["unit", "qty", "quantity", "orders"]):
        return "units"
    return None


def _description(filename: str, target_col: str | None, entity_cols: list[str], date_col: str | None) -> str:
    if target_col and date_col:
        entity = " by " + ", ".join(entity_cols) if entity_cols else ""
        return f"{Path(filename).name} contains time-series {target_col}{entity}."
    return f"{Path(filename).name} does not look like a complete demand time series."


def _questions(items: list[str], target_col: str | None, entity_cols: list[str]) -> list[str]:
    clean = [q.strip() for q in items if q and q.strip()][:3]
    if len(clean) == 3:
        return clean
    entity = "an entity" if entity_cols else "the dataset"
    target = _target_name(target_col)
    fallback = [
        f"What data am I working with?",
        f"Forecast {target} for {entity} over the next 7 days.",
        f"What recent trend or signals should I pay attention to?",
    ]
    return (clean + [q for q in fallback if q not in clean])[:3]


def _infer_region(df: pd.DataFrame) -> DatasetRegion | None:
    text_parts: list[str] = [" ".join(map(str, df.columns))]
    for col in df.columns[:12]:
        vals = df[col].dropna().astype(str).head(20).tolist()
        text_parts.extend(vals)
    text = " ".join(text_parts).lower()
    regions: list[tuple[tuple[str, ...], DatasetRegion]] = [
        (("germany", "deutschland", "berlin", "munich", "hamburg", " eur", "euro"), DatasetRegion(country_code="DE", country_name="Germany", lat=51.0, lon=9.0)),
        (("united states", "usa", "us", "new york", "california", " usd", "dollar"), DatasetRegion(country_code="US", country_name="United States", lat=39.8, lon=-98.6)),
        (("united kingdom", "uk", "london", "gbp", "pound"), DatasetRegion(country_code="GB", country_name="United Kingdom", lat=54.0, lon=-2.0)),
        (("india", "delhi", "mumbai", "bengaluru", "inr", "rupee"), DatasetRegion(country_code="IN", country_name="India", lat=22.0, lon=79.0)),
        (("france", "paris", "eur"), DatasetRegion(country_code="FR", country_name="France", lat=46.2, lon=2.2)),
    ]
    for keys, region in regions:
        if any(k in text for k in keys):
            return region
    return None

from __future__ import annotations

from datetime import date

import httpx

from app.core.config import settings

# Signal enrichers. Each joins onto any series by date and degrades to a mock when
# the network/key is unavailable, so the Sensing agent always has something to work
# with. These add CONTEXT signals, never the forecast target.

_TIMEOUT = 8.0


async def fetch_holidays(year: int, country: str = "DE") -> list[dict]:
    """Public holidays via Nager.Date (no key). Rossmann is German → default DE."""
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(url)
            r.raise_for_status()
            return [{"date": h["date"], "name": h["localName"]} for h in r.json()]
    except Exception:
        return _mock_holidays(year)


def _mock_holidays(year: int) -> list[dict]:
    return [
        {"date": f"{year}-01-01", "name": "New Year (mock)"},
        {"date": f"{year}-12-25", "name": "Christmas (mock)"},
    ]


async def fetch_weather_summary(
    start: date, end: date, lat: float = 51.0, lon: float = 9.0
) -> dict:
    """Open-Meteo historical weather (no key). Default lat/lon ~ central Germany."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            temps = r.json().get("daily", {}).get("temperature_2m_mean", [])
            temps = [t for t in temps if t is not None]
            if not temps:
                return _mock_weather()
            return {
                "avg_temp_c": round(sum(temps) / len(temps), 1),
                "min_temp_c": round(min(temps), 1),
                "max_temp_c": round(max(temps), 1),
                "source": "open-meteo",
            }
    except Exception:
        return _mock_weather()


def _mock_weather() -> dict:
    return {"avg_temp_c": 12.0, "min_temp_c": 4.0, "max_temp_c": 20.0, "source": "mock"}


async def fetch_macro_signal() -> dict:
    """Optional macro context via FRED (consumer sentiment). Mocked without a key.

    Note: illustrative for Rossmann (German, 2013–2015) since FRED is US/current.
    The architecture treats macro as an optional context signal, not a forecast driver.
    """
    if not settings.has_fred:
        return {"series": "UMCSENT", "value": 90.0, "source": "mock"}
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "UMCSENT",
        "api_key": settings.fred_api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            val = float(obs[0]["value"]) if obs else 90.0
            return {"series": "UMCSENT", "value": val, "source": "fred"}
    except Exception:
        return {"series": "UMCSENT", "value": 90.0, "source": "mock"}

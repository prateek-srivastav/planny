"""Destination weather via Open-Meteo (keyless).

The forecast API accepts dates up to today + 14. Trips starting later get climatology for the
same calendar window over the last three full years, flagged `basis="historical_climatology"`.
"""

import asyncio
from datetime import date, timedelta
from statistics import mean

import httpx

from app.graph.state import ConditionsSummary, DailyCondition
from app.tools.collector import ToolFailure
from app.tools.geo import Place

# Open-Meteo accepts end_date up to today + 14 (it answers 400 beyond that).
FORECAST_HORIZON_DAYS = 14
RAINY_DAY_MM = 1.0
CLIMATOLOGY_YEARS = 3
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"


def _shift_year(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year - years, day=28)


def _rows(payload: dict) -> list[DailyCondition]:
    daily = payload.get("daily") or {}
    times = daily.get("time") or []

    def col(name: str) -> list:
        return daily.get(name) or [None] * len(times)

    return [
        DailyCondition(
            date=date.fromisoformat(t),
            temp_max_c=hi,
            temp_min_c=lo,
            precipitation_mm=rain,
            precipitation_probability=prob,
        )
        for t, hi, lo, rain, prob in zip(
            times,
            col("temperature_2m_max"),
            col("temperature_2m_min"),
            col("precipitation_sum"),
            col("precipitation_probability_max"),
            strict=True,
        )
    ]


def _avg(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 1) if vals else None


def summarize(rows: list[DailyCondition]) -> dict:
    rain = [r.precipitation_mm for r in rows if r.precipitation_mm is not None]
    return {
        "avg_high_c": _avg([r.temp_max_c for r in rows]),
        "avg_low_c": _avg([r.temp_min_c for r in rows]),
        "avg_daily_precip_mm": _avg(rain),
        "rainy_day_fraction": round(sum(v >= RAINY_DAY_MM for v in rain) / len(rain), 2)
        if rain
        else None,
    }


async def _get(client: httpx.AsyncClient, url: str, params: dict) -> dict:
    resp = await client.get(url, params=params)
    if resp.status_code >= 400:
        raise ToolFailure(f"destination_conditions: open-meteo returned HTTP {resp.status_code}")
    return resp.json()


async def fetch_conditions(
    client: httpx.AsyncClient, place: Place, start: date, end: date, today: date | None = None
) -> ConditionsSummary:
    today = today or date.today()
    base = {
        "latitude": place.lat,
        "longitude": place.lon,
        "daily": DAILY_VARS,
        "timezone": "auto",
    }
    horizon = today + timedelta(days=FORECAST_HORIZON_DAYS)

    if start <= horizon:
        window_end = min(end, horizon)
        payload = await _get(
            client,
            "https://api.open-meteo.com/v1/forecast",
            {**base, "start_date": start.isoformat(), "end_date": window_end.isoformat()},
        )
        rows = _rows(payload)
        note = "Daily forecast."
        if window_end < end:
            note = (
                f"Forecast covers {start} to {window_end} only (forecast horizon); "
                f"days after that are unforecast."
            )
        return ConditionsSummary(
            basis="forecast",
            location=place.name,
            latitude=place.lat,
            longitude=place.lon,
            daily=rows,
            note=note,
            **summarize(rows),
        )

    years = [today.year - i for i in range(1, CLIMATOLOGY_YEARS + 1)]
    payloads = await asyncio.gather(
        *(
            _get(
                client,
                "https://archive-api.open-meteo.com/v1/archive",
                {
                    **base,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "start_date": _shift_year(start, start.year - y).isoformat(),
                    "end_date": _shift_year(end, start.year - y).isoformat(),
                },
            )
            for y in years
        )
    )
    rows = [r for p in payloads for r in _rows(p)]
    return ConditionsSummary(
        basis="historical_climatology",
        location=place.name,
        latitude=place.lat,
        longitude=place.lon,
        note=(
            f"Trip is more than {FORECAST_HORIZON_DAYS} days away, so no forecast exists. "
            f"Averages are for the same dates in {min(years)}-{max(years)}."
        ),
        **summarize(rows),
    )

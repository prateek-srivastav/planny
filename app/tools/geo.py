"""Geocoding plus the geometry behind `geo_cluster_and_route`.

Destinations and attractions are resolved with Nominatim (OpenStreetMap), which ranks results by
importance and knows regions, states, islands and countries ("Goa", "Bali", "Swiss Alps"), not only
cities. Its usage policy allows 1 request/second, so all lookups share one throttled, cached client.
Open-Meteo's city geocoder is the fallback for destinations when Nominatim is unavailable.
"""

import asyncio
import math
import time
from dataclasses import dataclass

import httpx

from app.tools.collector import ToolFailure


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    country: str | None = None


async def geocode_city(client: httpx.AsyncClient, query: str) -> Place | None:
    """Open-Meteo city search. Cities only, top hit by name; used as a fallback."""
    resp = await client.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": query, "count": 1, "language": "en", "format": "json"},
    )
    if resp.status_code >= 400:
        raise ToolFailure(f"geocoding: open-meteo returned HTTP {resp.status_code}")
    results = resp.json().get("results") or []
    if not results:
        return None
    r = results[0]
    return Place(r["name"], r["latitude"], r["longitude"], r.get("country"))


class Geocoder:
    """Nominatim client: throttled to its 1 request/second policy, cached per process."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        min_interval: float = 1.05,
        fallback: bool = True,
    ):
        self._client, self._base = client, base_url.rstrip("/")
        self._min_interval = min_interval
        self._fallback = fallback
        self._cache: dict[str, dict | None] = {}
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def _search(self, query: str, tool: str) -> dict | None:
        key = query.lower()
        if key in self._cache:
            return self._cache[key]
        async with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            resp = await self._client.get(
                f"{self._base}/search",
                params={
                    "q": query,
                    "format": "jsonv2",
                    "limit": 1,
                    "addressdetails": 1,
                    "accept-language": "en",
                },
            )
            self._last = time.monotonic()
        if resp.status_code >= 400:
            raise ToolFailure(f"{tool}: nominatim returned HTTP {resp.status_code}")
        hits = resp.json()
        hit = hits[0] if hits else None
        self._cache[key] = hit
        return hit

    async def destination(self, query: str) -> Place | None:
        """Resolve a travel destination (city, region, island or country)."""
        try:
            hit = await self._search(query, "geocoding")
        except (ToolFailure, httpx.HTTPError):
            if not self._fallback:
                raise
            return await geocode_city(self._client, query)
        if hit is None:
            return await geocode_city(self._client, query) if self._fallback else None
        name = hit.get("name") or hit["display_name"].split(",")[0]
        country = (hit.get("address") or {}).get("country")
        return Place(name, float(hit["lat"]), float(hit["lon"]), country)

    async def locate(self, name: str, near: str) -> tuple[float, float] | None:
        """Coordinates of a named attraction near a destination."""
        hit = await self._search(f"{name}, {near}", "geo_cluster_and_route")
        return (float(hit["lat"]), float(hit["lon"])) if hit else None


# --------------------------------------------------------------------------- geometry


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def travel_minutes(km: float) -> int:
    """Rough door-to-door estimate: walk short hops, otherwise urban transit with a wait."""
    if km <= 1.5:
        return round(km / 4.5 * 60)
    return round(km / 18 * 60 + 5)


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def cluster(points: list[tuple[float, float]], k: int, iterations: int = 15) -> list[list[int]]:
    """Group point indices into at most k geographic clusters (deterministic k-means).

    Seeds with farthest-point selection instead of random, so the same input always yields the
    same itinerary skeleton.
    """
    k = max(1, min(k, len(points)))
    centre = _centroid(points)
    seeds = [max(range(len(points)), key=lambda i: haversine_km(points[i], centre))]
    while len(seeds) < k:
        seeds.append(
            max(
                (i for i in range(len(points)) if i not in seeds),
                key=lambda i: min(haversine_km(points[i], points[s]) for s in seeds),
            )
        )
    centroids = [points[s] for s in seeds]
    groups: list[list[int]] = []
    for _ in range(iterations):
        groups = [[] for _ in range(k)]
        for i, p in enumerate(points):
            groups[min(range(k), key=lambda c: haversine_km(p, centroids[c]))].append(i)
        new = [
            _centroid([points[i] for i in g]) if g else centroids[c] for c, g in enumerate(groups)
        ]
        if new == centroids:
            break
        centroids = new
    return [g for g in groups if g]


def order_route(indices: list[int], points: list[tuple[float, float]]) -> list[int]:
    """Nearest-neighbour walk starting from the stop furthest from the group's centre."""
    if len(indices) <= 2:
        return indices
    centre = _centroid([points[i] for i in indices])
    current = max(indices, key=lambda i: haversine_km(points[i], centre))
    route, left = [current], set(indices) - {current}
    while left:
        current = min(left, key=lambda i: haversine_km(points[current], points[i]))
        route.append(current)
        left.remove(current)
    return route


def plan_days(named_points: dict[str, tuple[float, float]], num_days: int) -> list[dict]:
    names = list(named_points)
    points = [named_points[n] for n in names]
    days = []
    for n, group in enumerate(cluster(points, num_days), start=1):
        route = order_route(group, points)
        legs = [
            {
                "from": names[a],
                "to": names[b],
                "km": round(haversine_km(points[a], points[b]), 2),
                "minutes": travel_minutes(haversine_km(points[a], points[b])),
            }
            for a, b in zip(route, route[1:], strict=False)
        ]
        days.append(
            {
                "group": n,
                "stops": [
                    {"name": names[i], "lat": points[i][0], "lon": points[i][1]} for i in route
                ],
                "legs": legs,
                "total_travel_minutes": sum(leg["minutes"] for leg in legs),
            }
        )
    return days

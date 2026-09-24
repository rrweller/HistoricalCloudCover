"""Settings model, grid construction and cache-path rules.

Cache compatibility note
------------------------
The original script wrote one JSON file per grid point to
``.cache/cache_{lat}_{lon}.json`` holding nightly mean cloud cover derived with a
*fixed* local-time frame (Europe/London) and a *fixed* night window (18:00 ->
06:00).  Those derived values cannot be re-cut for a different night window, so
the cache is namespaced by the night definition: the legacy combination keeps
writing to the flat ``.cache/`` directory, every other combination gets its own
subdirectory.  That preserves the existing corpus while letting the UI expose
the night definition as a real setting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime, date

import numpy as np

# How the nightly window is defined, from strictest to loosest:
#   astronomical  sun more than 18 degrees below the horizon (true darkness)
#   night         sun below the horizon at all (sunset to sunrise)
#   all           every hour of the day
# The middle one exists because above ~48.5 degrees the astronomical definition
# yields nothing at midsummer, so high latitudes need a usable alternative.
NIGHT_MODES = ("astronomical", "night", "all")

CACHE_ROOT = ".cache"
FAST_DIRNAME = "_fast"  # binary mirror of the JSON caches, for quick reloads

MIN_DATE = date(1940, 1, 1)  # Open-Meteo ERA5 archive start


def _worker_ceiling() -> int:
    """The public tier refuses bursts above ~8; a self-hosted one has no limit."""
    from .fetch import is_local_archive  # lazy: config must not import fetch at load

    return 64 if is_local_archive() else 12


class SettingsError(ValueError):
    """Raised when incoming settings cannot produce a valid run."""


@dataclass
class Settings:
    """Everything that determines *which numbers get fetched*.

    View-time choices (metric, colormap, season filter, ...) deliberately live
    outside this object: they are re-derived from cached nightly values without
    touching the network.
    """

    lat_min: float = 35.0
    lat_max: float = 70.0
    lon_min: float = -10.0
    lon_max: float = 40.0
    grid_resolution: int = 20

    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"

    night_mode: str = "astronomical"

    # Open-Meteo rejects bursts of ~8 simultaneous requests with "Too many
    # concurrent requests", so the pool stays small by default.  One worker
    # holds at most one request in flight, so this is the concurrency cap.
    workers: int = 4

    # ---------------------------------------------------------------- parsing

    @classmethod
    def from_dict(cls, raw: dict) -> "Settings":
        def num(key, cast, default):
            val = raw.get(key, default)
            if val is None or val == "":
                return default
            try:
                return cast(val)
            except (TypeError, ValueError):
                raise SettingsError(f"'{key}' is not a valid number")

        s = cls(
            lat_min=num("lat_min", float, 35.0),
            lat_max=num("lat_max", float, 70.0),
            lon_min=num("lon_min", float, -10.0),
            lon_max=num("lon_max", float, 40.0),
            grid_resolution=num("grid_resolution", int, 20),
            start_date=str(raw.get("start_date") or "2020-01-01"),
            end_date=str(raw.get("end_date") or "2024-12-31"),
            night_mode=str(raw.get("night_mode") or "astronomical"),
            workers=num("workers", int, 4),
        )
        s.validate()
        return s

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.lat_min >= self.lat_max:
            raise SettingsError("North bound must be greater than south bound")
        if self.lon_min >= self.lon_max:
            raise SettingsError("East bound must be greater than west bound")
        if not (-90 <= self.lat_min < self.lat_max <= 90):
            raise SettingsError("Latitudes must be between -90 and 90")
        if not (-180 <= self.lon_min < self.lon_max <= 180):
            raise SettingsError("Longitudes must be between -180 and 180")
        if not (2 <= self.grid_resolution <= 200):
            raise SettingsError("Grid resolution must be between 2 and 200")
        if self.night_mode not in NIGHT_MODES:
            raise SettingsError("Night mode must be 'astronomical' or 'all'")
        ceiling = _worker_ceiling()
        if not (1 <= self.workers <= ceiling):
            raise SettingsError(f"Workers must be between 1 and {ceiling}")

        start, end = self.start, self.end
        if start > end:
            raise SettingsError("Start date must be on or before end date")
        if start < MIN_DATE:
            raise SettingsError("Start date must be 1940-01-01 or later")

    # ------------------------------------------------------------------ dates

    @property
    def start(self) -> date:
        try:
            return datetime.fromisoformat(self.start_date).date()
        except ValueError:
            raise SettingsError(f"Bad start date '{self.start_date}'")

    @property
    def end(self) -> date:
        try:
            return datetime.fromisoformat(self.end_date).date()
        except ValueError:
            raise SettingsError(f"Bad end date '{self.end_date}'")

    def year_intervals(self) -> list[tuple[str, str]]:
        """Calendar-year request windows clipped to the run range.

        Chunking by year keeps request URLs identical to the ones the original
        script issued, so the existing ``.requests_http_cache`` still hits.
        """
        out = []
        for year in range(self.start.year, self.end.year + 1):
            y0 = max(date(year, 1, 1), self.start)
            y1 = min(date(year, 12, 31), self.end)
            out.append((y0.isoformat(), y1.isoformat()))
        return out

    def day_span(self) -> tuple[int, int]:
        """Run range as inclusive days-since-epoch, for numpy filtering."""
        epoch = date(1970, 1, 1)
        return (self.start - epoch).days, (self.end - epoch).days

    # ------------------------------------------------------------------- grid

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)

    def grid_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample axes spanning the box exactly — the outer ring of points sits
        on the box edges, with nothing sampled outside it."""
        lats = np.linspace(self.lat_min, self.lat_max, self.grid_resolution)
        lons = np.linspace(self.lon_min, self.lon_max, self.grid_resolution)
        return lats, lons

    def grid_points(self) -> list[tuple[float, float]]:
        """Sample points in the same (lat-major) order the original script used."""
        lats, lons = self.grid_axes()
        return [(float(la), float(lo)) for la in lats for lo in lons]

    @property
    def point_count(self) -> int:
        return self.grid_resolution * self.grid_resolution

    # ------------------------------------------------------------------ cache

    @property
    def night_slug(self) -> str:
        return {"astronomical": "astro", "night": "night", "all": "all24"}[self.night_mode]

    @property
    def cache_dir(self) -> str:
        return os.path.join(CACHE_ROOT, self.night_slug)

    @property
    def fast_cache_dir(self) -> str:
        return os.path.join(CACHE_ROOT, FAST_DIRNAME, self.night_slug)

    def cache_path(self, lat: float, lon: float) -> str:
        return os.path.join(self.cache_dir, f"cache_{float(lat)}_{float(lon)}.json")

    def fast_cache_path(self, lat: float, lon: float) -> str:
        return os.path.join(self.fast_cache_dir, f"{float(lat)}_{float(lon)}.npz")

    def ensure_cache_dirs(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.fast_cache_dir, exist_ok=True)

    # -------------------------------------------------------------- describing

    def night_label(self) -> str:
        return {
            "astronomical": "astronomical night (sun below -18°)",
            "night": "sun below the horizon",
            "all": "all 24 hours",
        }[self.night_mode]

"""Turning nightly values into map-ready fields, metrics and rankings.

Everything here is pure numpy over the arrays a run produced, so switching
metric, season or threshold is a millisecond-scale re-aggregation rather than
another pass over the network.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata

from .config import Settings

SEASONS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}

METRICS = {
    "mean": {
        "label": "Mean cloud cover",
        "unit": "%",
        "help": "Average cloud cover across every night in the range.",
    },
}

MERC_LIMIT = 85.05112878


@dataclass
class RunResult:
    """Flat, column-oriented store of every night at every point."""

    settings: Settings
    lats: np.ndarray  # (P,) point latitudes
    lons: np.ndarray  # (P,) point longitudes
    pt: np.ndarray  # (N,) index into lats/lons
    day: np.ndarray  # (N,) days since epoch
    val: np.ndarray  # (N,) nightly mean cloud cover, %
    month: np.ndarray  # (N,) 1-12
    failures: list

    @property
    def n_points(self) -> int:
        return int(self.lats.size)

    @property
    def n_nights(self) -> int:
        return int(self.val.size)


def build_result(settings: Settings, points, per_point, failures) -> RunResult:
    """Assemble a RunResult from ``[(days, means), ...]`` in point order."""
    lats = np.array([p[0] for p in points], dtype=np.float64)
    lons = np.array([p[1] for p in points], dtype=np.float64)

    first, last = settings.day_span()
    pt_chunks, day_chunks, val_chunks = [], [], []
    for idx, (days, vals) in enumerate(per_point):
        if days is None or days.size == 0:
            continue
        keep = (days >= first) & (days <= last) & np.isfinite(vals)
        if not keep.any():
            continue
        d = days[keep]
        pt_chunks.append(np.full(d.size, idx, dtype=np.int32))
        day_chunks.append(d.astype(np.int32))
        val_chunks.append(vals[keep].astype(np.float32))

    if pt_chunks:
        pt = np.concatenate(pt_chunks)
        day = np.concatenate(day_chunks)
        val = np.concatenate(val_chunks)
    else:
        pt = np.empty(0, np.int32)
        day = np.empty(0, np.int32)
        val = np.empty(0, np.float32)

    month = (
        (day.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12) + 1
    ).astype(np.int8)

    return RunResult(settings, lats, lons, pt, day, val, month, list(failures))


# ------------------------------------------------------------------ filtering


def month_filter(period: str, custom_months=None) -> list[int]:
    period = (period or "all").lower()
    if period in SEASONS:
        return list(SEASONS[period])
    if period == "custom" and custom_months:
        months = sorted({int(m) for m in custom_months if 1 <= int(m) <= 12})
        return months or list(range(1, 13))
    return list(range(1, 13))


# ---------------------------------------------------------------- aggregation


def aggregate(result: RunResult, metric: str, months, threshold: float):
    """Collapse nightly values to one number per grid point.

    Returns ``(values, night_counts)``, both length P, NaN where a point has no
    nights matching the filter.
    """
    n_points = result.n_points
    if result.n_nights == 0:
        return np.full(n_points, np.nan), np.zeros(n_points, np.int64)

    months = list(months)
    if len(months) == 12:
        pt, val = result.pt, result.val
    else:
        keep = np.isin(result.month, np.asarray(months, np.int8))
        pt, val = result.pt[keep], result.val[keep]

    counts = np.bincount(pt, minlength=n_points).astype(np.int64)
    if val.size == 0:
        return np.full(n_points, np.nan), counts

    if metric == "median":
        values = _grouped_median(pt, val, n_points)
    elif metric in ("clear", "cloudy"):
        hit = (val < threshold) if metric == "clear" else (val > threshold)
        sums = np.bincount(pt, weights=hit.astype(np.float64), minlength=n_points)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(counts > 0, 100.0 * sums / np.maximum(counts, 1), np.nan)
    else:  # mean
        sums = np.bincount(pt, weights=val.astype(np.float64), minlength=n_points)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    values = np.where(counts > 0, values, np.nan)
    return values, counts


def _grouped_median(pt, val, n_points):
    order = np.lexsort((val, pt))
    sorted_pt = pt[order]
    sorted_val = val[order]
    edges = np.searchsorted(sorted_pt, np.arange(n_points + 1))
    out = np.full(n_points, np.nan)
    for i in range(n_points):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            out[i] = np.median(sorted_val[lo:hi])
    return out


# -------------------------------------------------------------- interpolation


def _merc_y(lat: float) -> float:
    lat = max(-MERC_LIMIT, min(MERC_LIMIT, lat))
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _inv_merc_y(y):
    return np.degrees(2.0 * np.arctan(np.exp(y)) - math.pi / 2)


def interpolate(
    lats,
    lons,
    values,
    bounds,
    width: int,
    height: int,
    method: str = "cubic",
):
    """Interpolate scattered point values onto a raster for the web overlay.

    Rows are spaced evenly in Web Mercator Y rather than in latitude, so Leaflet
    (which places an image overlay linearly in projected space) lands every pixel
    exactly where it belongs instead of smearing the field toward the poles.
    """
    lat_min, lat_max, lon_min, lon_max = bounds
    ok = np.isfinite(values)
    if ok.sum() < 4:
        return None, None, None

    pts = np.column_stack([np.asarray(lons)[ok], np.asarray(lats)[ok]])
    vals = np.asarray(values)[ok]

    y_top, y_bot = _merc_y(lat_max), _merc_y(lat_min)
    row_lats = _inv_merc_y(np.linspace(y_top, y_bot, height))
    col_lons = np.linspace(lon_min, lon_max, width)
    lon_mesh, lat_mesh = np.meshgrid(col_lons, row_lats)

    if method not in ("cubic", "linear", "nearest"):
        method = "cubic"
    grid = griddata(pts, vals, (lon_mesh, lat_mesh), method=method)

    if grid is not None and method != "nearest":
        # Cubic and linear both leave NaN outside the convex hull; backfill with
        # nearest so the raster has no ragged transparent fringe at the edges.
        holes = ~np.isfinite(grid)
        if holes.any():
            fill = griddata(pts, vals, (lon_mesh[holes], lat_mesh[holes]), method="nearest")
            grid[holes] = fill

    return grid, row_lats, col_lons


def encode_grid(grid) -> str:
    """Base64 float32, row-major, north row first — decoded straight into a
    Float32Array in the browser."""
    if grid is None:
        return ""
    return base64.b64encode(np.ascontiguousarray(grid, dtype=np.float32).tobytes()).decode()


def point_payload(result: RunResult, values, counts, max_points: int = 4000):
    """Sampled points for the map's dot layer and CSV export."""
    out = []
    step = max(1, result.n_points // max_points)
    for i in range(0, result.n_points, step):
        if not np.isfinite(values[i]):
            continue
        out.append(
            [
                round(float(result.lats[i]), 4),
                round(float(result.lons[i]), 4),
                round(float(values[i]), 2),
                int(counts[i]),
            ]
        )
    return out

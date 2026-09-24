"""Open-Meteo archive access and the per-grid-point nightly cache.

One grid point resolves to one JSON file of nightly mean cloud cover, in the
same format the original script wrote, so the existing corpus under ``.cache/``
keeps working.  Two things are layered on top:

*   a binary ``.npz`` mirror, because parsing thousands of JSON files dominates
    the runtime of an otherwise fully cached run;
*   ``n`` (hours contributing to a night) alongside ``average``, so nights that
    straddle two yearly requests merge correctly instead of one half-night
    silently overwriting the other.
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pandas as pd
import openmeteo_requests
import requests
import requests_cache
from retry_requests import retry

from . import diagnostics
from .config import Settings
from .solar import ASTRONOMICAL_DEPRESSION, HORIZON_DEPRESSION, sun_below

PUBLIC_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Point this at a self-hosted Open-Meteo instance to escape the public tier's
# rate limits. The container serves the identical API, so nothing else changes.
#   docker compose up -d   ->   http://127.0.0.1:8080/v1/archive
ARCHIVE_URL = os.environ.get("CLOUDCOVER_ARCHIVE_URL", PUBLIC_ARCHIVE_URL)


def is_local_archive() -> bool:
    """True when we are talking to something other than the public tier.

    Self-hosted instances have no quota, so the daily-budget warning and the
    small worker ceiling both stop applying.
    """
    return ARCHIVE_URL != PUBLIC_ARCHIVE_URL
HTTP_CACHE = ".requests_http_cache"

MAX_ATTEMPTS = 5
FIRST_BACKOFF = 5.0
MAX_BACKOFF = 90.0

# Open-Meteo's free tier allowance. Used only to warn before a run that cannot
# finish in one day — nothing enforces it locally.
DAILY_REQUEST_LIMIT = 10_000

_local = threading.local()
_io_lock = threading.Lock()


class Cancelled(Exception):
    """Raised inside a worker when the owning job has been cancelled."""


# --------------------------------------------------------------------- session


_http_cache_broken = False


def _session():
    """One cached+retrying HTTP session per thread.

    ``requests_cache`` keeps SQLite connections thread-local internally, but the
    session object still carries per-connection state, so we do not share one
    across the pool.

    If the caching layer has been found broken (see ``disable_http_cache``) this
    hands back a plain session instead: losing the response cache only costs
    speed, whereas failing every request costs the whole run.
    """
    sess = getattr(_local, "session", None)
    if sess is None:
        if _http_cache_broken:
            sess = retry(requests.Session(), retries=3, backoff_factor=0.3)
        else:
            cached = requests_cache.CachedSession(HTTP_CACHE, expire_after=-1)
            sess = retry(cached, retries=3, backoff_factor=0.3)
        _local.session = sess
    return sess


def disable_http_cache(reason: str = "") -> bool:
    """Stop using requests_cache in this process. True if this call did it."""
    global _http_cache_broken
    if _http_cache_broken:
        return False
    _http_cache_broken = True
    _local.session = None
    return True


# ------------------------------------------------------------ night bucketing


def night_days(times_utc: pd.DatetimeIndex, settings: Settings, lat: float, lon: float):
    """Map each hourly sample to the night it belongs to.

    Returns ``(night_day, keep)``. ``keep`` masks the hours that count; in
    astronomical mode that is the hours the sun sits below -18 degrees at this
    exact point and date, which is why latitude matters and why high-latitude
    summers legitimately contribute nothing.

    Nights are indexed by the evening they began, so a span crossing midnight
    stays one night. The split is at local solar noon — the far side of the day
    from the darkness — so it never lands inside a night.
    """
    # Mean solar time at this longitude: no timezone needed, and it keeps the
    # day boundary in the same place as the sun.
    solar = (times_utc + pd.Timedelta(hours=float(lon) / 15.0)).tz_localize(None)
    days = solar.normalize().to_numpy().astype("datetime64[D]").astype(np.int32)

    if settings.night_mode == "all":
        return days, np.ones(days.shape, dtype=bool)

    hours = np.asarray(solar.hour, dtype=np.int16)
    nights = np.where(hours >= 12, days, days - 1).astype(np.int32)
    depression = (ASTRONOMICAL_DEPRESSION if settings.night_mode == "astronomical"
                  else HORIZON_DEPRESSION)
    return nights, sun_below(times_utc, lat, lon, depression)


def _group_nights(nights: np.ndarray, values: np.ndarray, keep: np.ndarray):
    """Collapse hourly values into per-night (sum, count)."""
    nights = nights[keep]
    values = values[keep]
    finite = np.isfinite(values)
    nights = nights[finite]
    values = values[finite].astype(np.float64)
    if nights.size == 0:
        return (
            np.empty(0, np.int32),
            np.empty(0, np.float64),
            np.empty(0, np.int32),
        )
    uniq, inv = np.unique(nights, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=uniq.size)
    counts = np.bincount(inv, minlength=uniq.size)
    return uniq.astype(np.int32), sums, counts.astype(np.int32)


def _merge(parts):
    """Combine several (days, sums, counts) triples, adding overlaps."""
    parts = [p for p in parts if p[0].size]
    if not parts:
        return (np.empty(0, np.int32), np.empty(0, np.float64), np.empty(0, np.int32))
    days = np.concatenate([p[0] for p in parts])
    sums = np.concatenate([p[1] for p in parts])
    counts = np.concatenate([p[2] for p in parts])
    uniq, inv = np.unique(days, return_inverse=True)
    return (
        uniq.astype(np.int32),
        np.bincount(inv, weights=sums, minlength=uniq.size),
        np.bincount(inv, weights=counts, minlength=uniq.size).astype(np.int32),
    )


# ---------------------------------------------------------------- cache layer


def _iso(day: int) -> str:
    return str(np.datetime64(int(day), "D"))


def _day_num(iso: str) -> int | None:
    try:
        return int(np.datetime64(iso, "D").astype(int))
    except Exception:
        return None


def _read_fast(path: str):
    try:
        with np.load(path) as z:
            return z["day"], z["sum"], z["count"], set(int(y) for y in z["years"])
    except Exception:
        return None


def _write_fast(path: str, days, sums, counts, years) -> None:
    # Written through a file handle so savez does not append its own extension.
    tmp = f"{path}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                day=np.asarray(days, np.int32),
                sum=np.asarray(sums, np.float64),
                count=np.asarray(counts, np.int32),
                years=np.asarray(sorted(years), np.int32),
            )
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _read_json(path: str):
    """Load a legacy/current per-point JSON cache into arrays."""
    with open(path, "r") as fh:
        blob = json.load(fh)

    years = set(int(y) for y in blob.get("_years", []) if str(y).isdigit())
    days, sums, counts = [], [], []
    for key, rec in blob.items():
        if key.startswith("_") or not isinstance(rec, dict):
            continue
        day = _day_num(key)
        if day is None:
            continue
        avg = rec.get("average")
        if avg is None:
            continue
        try:
            avg = float(avg)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(avg):
            continue
        # Legacy records have no hour count; treat them as a single weighted
        # observation so the stored mean is preserved exactly.
        n = int(rec.get("n") or 1)
        days.append(day)
        sums.append(avg * n)
        counts.append(n)
    order = np.argsort(days) if days else np.empty(0, int)
    return (
        np.asarray(days, np.int32)[order],
        np.asarray(sums, np.float64)[order],
        np.asarray(counts, np.int32)[order],
        years,
    )


def _write_json(path: str, days, sums, counts, years) -> None:
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    blob = {}
    for day, mean, n in zip(days, means, counts):
        if not np.isfinite(mean):
            continue
        blob[_iso(day)] = {"average": float(mean), "n": int(n)}
    blob["_years"] = sorted(int(y) for y in years)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(blob, fh, indent=2)
    os.replace(tmp, path)


def load_point(settings: Settings, lat: float, lon: float):
    """Read whatever is already on disk for one point.

    Prefers the binary mirror, falls back to JSON and builds the mirror so the
    next run is fast.
    """
    fast_path = settings.fast_cache_path(lat, lon)
    json_path = settings.cache_path(lat, lon)

    json_mtime = os.path.getmtime(json_path) if os.path.exists(json_path) else None
    if os.path.exists(fast_path):
        if json_mtime is None or os.path.getmtime(fast_path) >= json_mtime:
            got = _read_fast(fast_path)
            if got is not None:
                return got

    if json_mtime is not None:
        try:
            days, sums, counts, years = _read_json(json_path)
        except Exception:
            days, sums, counts, years = (
                np.empty(0, np.int32),
                np.empty(0, np.float64),
                np.empty(0, np.int32),
                set(),
            )
        _write_fast(fast_path, days, sums, counts, years)
        return days, sums, counts, years

    return (
        np.empty(0, np.int32),
        np.empty(0, np.float64),
        np.empty(0, np.int32),
        set(),
    )


def save_point(settings: Settings, lat: float, lon: float, days, sums, counts, years):
    with _io_lock:
        settings.ensure_cache_dirs()
    _write_json(settings.cache_path(lat, lon), days, sums, counts, years)
    _write_fast(settings.fast_cache_path(lat, lon), days, sums, counts, years)


# ------------------------------------------------------------------- fetching


def _request_year(lat: float, lon: float, start: str, end: str):
    """One archive request.  Params match the original script's exactly so the
    pre-existing ``.requests_http_cache`` still serves them."""
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start,
        "end_date": end,
        "hourly": "cloud_cover",
    }
    try:
        client = openmeteo_requests.Client(session=_session())
        response = client.weather_api(ARCHIVE_URL, params=params)[0]
    except PERMANENT_ERRORS:
        # The response-cache layer is the usual suspect for this class of fault
        # (a requests_cache/requests version mismatch raises NameError on every
        # call). Drop it and try once uncached before giving up on the request.
        if not disable_http_cache():
            raise
        client = openmeteo_requests.Client(session=_session())
        response = client.weather_api(ARCHIVE_URL, params=params)[0]

    hourly = response.Hourly()
    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    values = hourly.Variables(0).ValuesAsNumpy()
    return times, np.asarray(values, dtype=np.float64)


def _sleep_cancellable(seconds: float, should_cancel) -> None:
    waited = 0.0
    while waited < seconds:
        if should_cancel():
            raise Cancelled()
        time.sleep(min(0.25, seconds - waited))
        waited += 0.25


# A broken dependency is not a transient failure. When requests_cache stopped
# working against requests 2.34, every call raised NameError and the backoff
# below patiently retried each one five times — turning a one-second breakage
# into a run that never finished. These exception types mean "the code or an
# installed package is wrong"; retrying cannot help, so we surface them at once.
PERMANENT_ERRORS = (NameError, ImportError, AttributeError, TypeError)


# Open-Meteo signals back-pressure several ways; "Too many concurrent requests"
# in particular carries none of the words you would guess, and missing it turns a
# polite backoff into every worker retrying in lockstep.
_BACKPRESSURE = ("limit", "429", "quota", "minutely", "hourly", "daily",
                 "concurrent", "too many", "throttl")


def is_backpressure(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(word in text for word in _BACKPRESSURE)


def fetch_year(lat, lon, start, end, should_cancel, on_retry=None, log=None):
    """Request a year with bounded exponential backoff.

    The original retried forever on any failure, which is fine for a script left
    running overnight but would wedge a web request, so failures are surfaced
    after a handful of attempts instead.
    """
    delay = FIRST_BACKOFF
    last = None
    for attempt in range(MAX_ATTEMPTS):
        if should_cancel():
            raise Cancelled()
        try:
            # Logged before the call and flushed, so if the process dies inside
            # the request this line names exactly what it was doing.
            if log:
                log.debug("REQ  %.4f,%.4f %s..%s try=%d", lat, lon, start, end, attempt + 1)
            began = time.monotonic()
            result = _request_year(lat, lon, start, end)
            if log:
                log.debug("OK   %.4f,%.4f %s..%s n=%d %.2fs",
                          lat, lon, start, end, len(result[1]), time.monotonic() - began)
            return result
        except Cancelled:
            raise
        except PERMANENT_ERRORS as exc:
            if log:
                log.error("BROKEN %.4f,%.4f %s..%s %s: %s — not retrying, this is a "
                          "code or dependency fault, not a network one",
                          lat, lon, start, end, type(exc).__name__, exc)
            raise RuntimeError(
                f"{type(exc).__name__}: {exc} — this looks like a broken or "
                f"mismatched Python package rather than a network problem"
            ) from exc
        except Exception as exc:  # network, rate limit, malformed payload
            last = exc
            if log:
                log.warning("FAIL %.4f,%.4f %s..%s try=%d %s: %s",
                            lat, lon, start, end, attempt + 1, type(exc).__name__, exc)
            if attempt == MAX_ATTEMPTS - 1:
                break
            if on_retry:
                on_retry(exc, attempt + 1, delay)
            # Back-pressure deserves a longer pause than a transient socket error.
            wait = delay * 3 if is_backpressure(exc) else delay
            _sleep_cancellable(min(wait, MAX_BACKOFF), should_cancel)
            delay = min(delay * 2, MAX_BACKOFF)
    raise RuntimeError(f"{lat},{lon} {start}..{end}: {last}") from last


def years_needed(settings: Settings, days, years_done) -> list:
    """Which calendar years still have to be requested for this point."""
    first, last = settings.day_span()
    wanted = set(range(first, last + 1))
    have = set(int(d) for d in days.tolist())
    missing = wanted - have

    out = []
    for y_start, y_end in settings.year_intervals():
        year = int(y_start[:4])
        if year in years_done:
            continue
        lo, hi = _day_num(y_start), _day_num(y_end)
        if any(lo <= d <= hi for d in missing):
            out.append((year, y_start, y_end))
    return out


def cached_point(settings: Settings, lat: float, lon: float):
    """Load one point from disk only — never touches the network.

    Returns ``(days, means, needs_fetch)``. Safe to run in threads, which is why
    the cache pass can stay in the web server's own process.
    """
    days, sums, counts, years_done = load_point(settings, lat, lon)
    return days, _means(sums, counts), bool(years_needed(settings, days, years_done))


def process_point(settings: Settings, lat: float, lon: float, should_cancel,
                  on_retry=None, log=None):
    """Return every nightly mean available for one point, fetching what is missing.

    Returns ``(days, means, fetched_years)``.
    """
    days, sums, counts, years_done = load_point(settings, lat, lon)
    needed = years_needed(settings, days, years_done)

    if not needed:
        return days, _means(sums, counts), 0

    parts = [(days, sums, counts)]
    for year, y_start, y_end in needed:
        times, values = fetch_year(lat, lon, y_start, y_end, should_cancel, on_retry, log)
        nights, keep = night_days(times, settings, lat, lon)
        parts.append(_group_nights(nights, values, keep))
        years_done.add(year)

    days, sums, counts = _merge(parts)
    if log:
        log.debug("SAVE %.4f,%.4f nights=%d years=%d", lat, lon, days.size, len(needed))
    save_point(settings, lat, lon, days, sums, counts, years_done)
    return days, _means(sums, counts), len(needed)


# ------------------------------------------------------- process-pool entry


# A single point may not monopolise a worker forever: this bounds how long a
# cancelled run takes to wind down, since a child cannot be interrupted midway.
POINT_TIME_BUDGET = 180.0


def run_point(task):
    """Top-level worker callable — must stay importable for the process pool.

    Returns ``(index, days, means, fetched_years, failure_message)`` and never
    raises, so one bad point cannot take the pool down with it.
    """
    settings, index, lat, lon, run_id = task
    diagnostics.enable_faulthandler("worker")
    log = diagnostics.run_logger(run_id)

    started = time.monotonic()

    def out_of_time():
        return (time.monotonic() - started) > POINT_TIME_BUDGET

    try:
        days, means, fetched = process_point(settings, lat, lon, out_of_time, None, log)
        return index, days, means, fetched, None
    except Cancelled:
        msg = f"{lat:.3f}, {lon:.3f}: gave up after {POINT_TIME_BUDGET:.0f}s"
        log.warning(msg)
        return index, None, None, 0, msg
    except Exception as exc:
        msg = f"{lat:.3f}, {lon:.3f}: {type(exc).__name__}: {exc}"
        log.warning("POINT FAILED %s", msg)
        return index, None, None, 0, msg


def _means(sums, counts):
    if sums.size == 0:
        return np.empty(0, np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return out.astype(np.float32)


# ------------------------------------------------------------------- estimate


def coverage_path(settings: Settings) -> str:
    return os.path.join(settings.fast_cache_dir, "coverage.json")


def read_coverage(settings: Settings) -> dict:
    """Per-namespace index of ``point -> [first_day, last_day, n_days]``.

    The estimate fires on every settings tweak, so it must not touch thousands
    of files; this one small file answers "what do we already have?" outright.
    """
    try:
        with open(coverage_path(settings), "r") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_coverage(settings: Settings, coverage: dict) -> None:
    if not coverage:
        return
    try:
        settings.ensure_cache_dirs()
        merged = read_coverage(settings)
        merged.update(coverage)
        path = coverage_path(settings)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(merged, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def point_key(lat: float, lon: float) -> str:
    return f"{float(lat)}_{float(lon)}"


def estimate(settings: Settings) -> dict:
    """Cheap pre-flight: how much of this run is already on disk?

    Answered from the namespace index where possible.  Points the index has not
    seen fall back to a file probe, and points with only the legacy JSON are
    reported as unverified rather than parsed — reading thousands of JSON files
    would take far longer than the estimate is worth.
    """
    points = settings.grid_points()
    first, last = settings.day_span()
    span = last - first + 1
    index = read_coverage(settings)

    ready = 0
    partial = 0
    unverified = 0
    missing = 0

    for lat, lon in points:
        entry = index.get(point_key(lat, lon))
        if entry and len(entry) == 3:
            d0, d1, n = entry
            if d0 <= first and d1 >= last and n >= (d1 - d0 + 1):
                ready += 1
            elif d1 >= first and d0 <= last:
                partial += 1
            else:
                missing += 1
            continue

        fast_path = settings.fast_cache_path(lat, lon)
        json_path = settings.cache_path(lat, lon)
        if os.path.exists(fast_path):
            got = _read_fast(fast_path)
            if got is not None:
                days = got[0]
                covered = int(((days >= first) & (days <= last)).sum())
                if covered >= span:
                    ready += 1
                elif covered > 0:
                    partial += 1
                else:
                    missing += 1
                continue
        if os.path.exists(json_path):
            unverified += 1
        else:
            missing += 1

    # The grid is a product of two axes, so points inside the un-buffered box
    # is just the product of the per-axis counts.
    axis_lats, axis_lons = settings.grid_axes()
    inside = int(((axis_lats >= settings.lat_min) & (axis_lats <= settings.lat_max)).sum()) * int(
        ((axis_lons >= settings.lon_min) & (axis_lons <= settings.lon_max)).sum()
    )

    to_fetch = partial + missing
    years = len(settings.year_intervals())
    max_requests = to_fetch * years
    # Rough wall-clock guess: one request per worker at a time. A local archive
    # answers from local disk or a warm chunk cache, so it is far quicker.
    per_request = 0.12 if is_local_archive() else 0.7
    eta = max_requests * per_request / max(1, settings.workers)
    return {
        "points": len(points),
        "points_inside": inside,
        "eta_seconds": round(eta),
        "over_daily_limit": (not is_local_archive()) and max_requests > DAILY_REQUEST_LIMIT,
        "daily_limit": DAILY_REQUEST_LIMIT,
        "ready": ready,
        "unverified": unverified,
        "partial": partial,
        "missing": missing,
        "to_fetch": to_fetch,
        "years": years,
        "max_requests": to_fetch * years,
        "nights": span,
    }


def archive_health() -> dict:
    """Ping the configured archive with a one-day, one-point request."""
    began = time.monotonic()
    try:
        times, _ = _request_year(52.0, 5.0, "2023-01-01", "2023-01-01")
        return {"ok": True, "url": ARCHIVE_URL, "local": is_local_archive(),
                "hours": len(times), "seconds": round(time.monotonic() - began, 3)}
    except Exception as exc:
        return {"ok": False, "url": ARCHIVE_URL, "local": is_local_archive(),
                "error": f"{type(exc).__name__}: {exc}"}

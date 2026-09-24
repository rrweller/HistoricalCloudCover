"""Web front end for the historical nighttime cloud-cover analysis.

Run it with:

    python app.py

then open http://127.0.0.1:5000 — pick an area on the map, set the date range and
night window, and the same pipeline the CLI uses runs in the background.
"""

from __future__ import annotations

import argparse
import io
import atexit
import multiprocessing
import os
import re
import signal
import sys
import threading
import webbrowser

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

from cloudcover import analysis, container, diagnostics, fetch, jobs
from cloudcover.config import Settings, SettingsError

# cloudcover.render pulls in matplotlib and cartopy, which take seconds to
# import. Worker processes re-import this module on spawn and never render, so
# it is loaded on demand inside the export route instead.

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Installed in every process, so a native crash leaves a Python stack behind
# whichever side of the pool it happens on.
diagnostics.enable_faulthandler(
    "server" if multiprocessing.current_process().name == "MainProcess" else "worker"
)

LOG_NAME_RE = re.compile(r"^[\w.\-]+\.log$")

def _settings_from_request() -> Settings:
    payload = request.get_json(silent=True) or {}
    return Settings.from_dict(payload.get("settings", payload))


def _view_params(payload: dict) -> dict:
    metric = str(payload.get("metric") or "mean")
    if metric not in analysis.METRICS:
        metric = "mean"
    try:
        threshold = float(payload.get("threshold", 25))
    except (TypeError, ValueError):
        threshold = 25.0
    threshold = min(100.0, max(0.0, threshold))
    resolution = int(payload.get("resolution") or 520)
    resolution = min(1200, max(120, resolution))
    method = payload.get("method") or "cubic"
    if method not in ("cubic", "linear", "nearest"):
        method = "cubic"
    raw_months = payload.get("months")
    if isinstance(raw_months, list) and raw_months:
        months = sorted({int(m) for m in raw_months if 1 <= int(m) <= 12})
    else:
        months = analysis.month_filter(payload.get("period") or "all")
    if not months:
        months = list(range(1, 13))

    return {
        "metric": metric,
        "threshold": threshold,
        "period": str(payload.get("period") or "all"),
        "label": str(payload.get("label") or "All year"),
        "months": months,
        "resolution": resolution,
        "method": method,
    }


def _compute_view(job: jobs.Job, params: dict):
    result = job.result
    months = params["months"]
    values, counts = analysis.aggregate(result, params["metric"], months, params["threshold"])
    return values, counts, months


# --------------------------------------------------------------------- routes


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/bootstrap")
def bootstrap():
    defaults = Settings()
    return jsonify(
        {
            "defaults": defaults.to_dict(),
            "archive": {
                "url": fetch.ARCHIVE_URL,
                "local": fetch.is_local_archive(),
                "max_workers": 64 if fetch.is_local_archive() else 12,
            },
            "night_modes": [
                {"key": "astronomical",
                 "label": "Astronomical night only",
                 "help": "Keeps only the hours the sun is more than 18° below the "
                         "horizon at each point, which varies with latitude and date. "
                         "Above about 48.5° north there is none at midsummer."},
                {"key": "night",
                 "label": "Sun below the horizon",
                 "help": "Every hour between sunset and sunrise. Looser than "
                         "astronomical night, so it still yields data through a "
                         "high-latitude summer where true darkness never arrives."},
                {"key": "all",
                 "label": "Full 24 hours",
                 "help": "Every hour of every day, day and night alike."},
            ],
            "metrics": {
                key: {"key": key, **meta} for key, meta in analysis.METRICS.items()
            },
        }
    )


@app.post("/api/estimate")
def estimate():
    try:
        settings = _settings_from_request()
    except SettingsError as exc:
        return jsonify({"error": str(exc)}), 400
    data = fetch.estimate(settings)
    data["night_label"] = settings.night_label()
    return jsonify(data)


@app.post("/api/run")
def run():
    try:
        settings = _settings_from_request()
    except SettingsError as exc:
        return jsonify({"error": str(exc)}), 400
    job = jobs.start(settings)
    return jsonify(job.snapshot())


@app.get("/api/job/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job.snapshot())


@app.post("/api/job/<job_id>/cancel")
def job_cancel(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    job.cancel()
    return jsonify(job.snapshot())


@app.post("/api/job/<job_id>/view")
def job_view(job_id):
    job = jobs.get(job_id)
    if job is None or job.result is None:
        return jsonify({"error": "No results for this run"}), 404

    params = _view_params(request.get_json(silent=True) or {})
    result = job.result
    values, counts, months = _compute_view(job, params)

    settings = result.settings
    lat_min, lat_max, lon_min, lon_max = settings.bounds
    aspect = max(0.15, (lat_max - lat_min) / max(lon_max - lon_min, 1e-6))
    width = params["resolution"]
    height = int(max(60, min(1400, width * aspect)))

    grid, _, _ = analysis.interpolate(
        result.lats, result.lons, values,
        (lat_min, lat_max, lon_min, lon_max),
        width, height, params["method"],
    )

    finite = values[np.isfinite(values)]

    return jsonify(
        {
            "job_id": job.id,
            "metric": {"key": params["metric"], **analysis.METRICS[params["metric"]]},
            "period": params["period"],
            "label": params["label"],
            "months": months,
            "empty": bool(not np.isfinite(values).any()),
            "grid": {
                "data": analysis.encode_grid(grid),
                "width": width,
                "height": height,
                "bounds": [lat_min, lat_max, lon_min, lon_max],
            },
            "box": [settings.lat_min, settings.lat_max, settings.lon_min, settings.lon_max],
            "data_min": round(float(finite.min()), 2) if finite.size else None,
            "data_max": round(float(finite.max()), 2) if finite.size else None,
            "points": analysis.point_payload(result, values, counts),
            "nights_per_point": int(counts.max()) if counts.size else 0,
            "night_label": settings.night_label(),
            "date_range": [settings.start_date, settings.end_date],
            "total_nights": result.n_nights,
            "failures": result.failures,
        }
    )


@app.post("/api/job/<job_id>/export.png")
def export_png(job_id):
    job = jobs.get(job_id)
    if job is None or job.result is None:
        return jsonify({"error": "No results for this run"}), 404

    from cloudcover import render  # heavy import, only needed here

    payload = request.get_json(silent=True) or {}
    params = _view_params(payload)
    result = job.result
    values, counts, _ = _compute_view(job, params)
    settings = result.settings
    meta = analysis.METRICS[params["metric"]]

    label = meta["label"]
    if params["metric"] in ("clear", "cloudy"):
        label = f"{label} (< {params['threshold']:g}% cloud)" if params["metric"] == "clear" \
            else f"{label} (> {params['threshold']:g}% cloud)"

    period_label = "" if params["label"] == "All year" else f" — {params['label']}"

    try:
        png = render.export_png(
            result.lats, result.lons, values,
            (settings.lat_min, settings.lat_max, settings.lon_min, settings.lon_max),
            title=f"{label}{period_label}",
            subtitle=f"{settings.start_date} to {settings.end_date} · night {settings.night_label()}",
            legend_label=f"{label} ({meta['unit']})",
            colormap=payload.get("colormap", "blue"),
            vmin=float(payload.get("vmin", 0)),
            vmax=float(payload.get("vmax", 100)),
            bands=int(payload.get("bands") or 10),
            show_points=bool(payload.get("show_points", True)),
            method=params["method"],
        )
    except Exception as exc:
        return jsonify({"error": f"Export failed: {exc}"}), 500

    slug = re.sub(r"[^a-z0-9]+", "_", params["label"].lower()).strip("_")
    name = f"cloudcover_{slug}.png"
    return send_file(io.BytesIO(png), mimetype="image/png",
                     as_attachment=True, download_name=name)


@app.get("/api/archive")
def archive_status():
    return jsonify(fetch.archive_health())


@app.get("/api/logs")
def list_logs():
    return jsonify({"dir": os.path.abspath(diagnostics.LOG_DIR),
                    "files": diagnostics.recent_logs()})


@app.get("/api/logs/<name>")
def read_log(name):
    """Serve a log file as plain text.

    Names are pattern-matched and the resolved path is confirmed to sit inside
    the log directory, so a crafted name cannot walk out of it.
    """
    if not LOG_NAME_RE.match(name):
        return Response("bad log name", mimetype="text/plain", status=400)
    root = os.path.abspath(diagnostics.LOG_DIR)
    path = os.path.abspath(os.path.join(root, name))
    if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
        return Response("no such log", mimetype="text/plain", status=404)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return Response(fh.read(), mimetype="text/plain")


def _prewarm_basemap():
    """Pull the Natural Earth shapefiles the PNG export needs into cartopy's cache.

    On a machine that has never rendered one, the first export otherwise blocks
    while cartopy downloads them, which reads as a hang. Doing it at startup
    overlaps the download with the user picking an area.
    """
    try:
        import cartopy.feature as cfeature

        for feature in (cfeature.COASTLINE, cfeature.BORDERS, cfeature.LAKES, cfeature.RIVERS):
            list(feature.geometries())
    except Exception:
        pass  # exports still work, they just pay the download once


def main():
    parser = argparse.ArgumentParser(description="Historical cloud cover explorer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--no-docker", action="store_true",
        help="Do not manage the archive container; use the public Open-Meteo tier.",
    )
    parser.add_argument(
        "--archive-url",
        help="Point at an archive endpoint you run yourself. Implies --no-docker.",
    )
    args = parser.parse_args()

    # Decide the archive before anything forks: worker processes inherit the
    # environment, and they must all agree on where the data comes from.
    started_container = False
    if args.archive_url:
        os.environ["CLOUDCOVER_ARCHIVE_URL"] = args.archive_url
        fetch.ARCHIVE_URL = args.archive_url
    elif not args.no_docker:
        try:
            started_container = container.start()
        except container.ContainerError as exc:
            print()
            print(exc)
            print()
            return 1
        os.environ["CLOUDCOVER_ARCHIVE_URL"] = container.LOCAL_ARCHIVE_URL
        fetch.ARCHIVE_URL = container.LOCAL_ARCHIVE_URL

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}"
    print()
    print(f"  Cloud Cover Explorer  ->  {url}")
    print(f"  Run + crash logs      ->  {os.path.abspath(diagnostics.LOG_DIR)}")
    tier = "self-hosted, no quota" if fetch.is_local_archive() else "public tier"
    print(f"  Archive               ->  {fetch.ARCHIVE_URL}  ({tier})")
    if started_container:
        print("  (container started by this process; it will stop on exit)")
    print()

    if not args.no_browser and not args.debug:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    threading.Thread(target=_prewarm_basemap, name="prewarm", daemon=True).start()

    # A container we started must come down however this process ends: Ctrl+C,
    # a closed terminal, taskkill, or a clean return. Python installs no SIGBREAK
    # handler of its own, so closing the console window would otherwise strand it.
    done = threading.Event()

    def cleanup():
        if started_container and not done.is_set():
            done.set()
            container.stop()

    def on_signal(signum, frame):
        cleanup()
        os._exit(0)

    if started_container:
        atexit.register(cleanup)
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, on_signal)
                except (ValueError, OSError):
                    pass

    try:
        # The reloader would duplicate in-flight jobs, so it stays off.
        app.run(host=args.host, port=args.port, debug=args.debug,
                threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())

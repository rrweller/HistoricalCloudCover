# Historical Cloud Cover

Find the clearest and cloudiest parts of a region, using Open-Meteo's ERA5
archive of hourly cloud cover.

Draw a box on a map, choose how many points to sample and which hours count, and
the tool averages cloud cover over every qualifying night in your date range and
interpolates the result into a map you can scrub through by season.

## Running the web interface

```
pip install -r requirements.txt
python app.py
```

Opens <http://127.0.0.1:5000>. Flags: `--port`, `--host`, `--no-browser`.

### How a session goes

1. **Pick an area.** Drag the box handles, drag the box itself, hit *Draw on map*
   to drag out a new one, or type bounds into the compass fields.
2. **Set the grid.** *Sample grid* is points per axis. The outer ring sits exactly
   on the box edges — nothing is sampled outside your selection.
3. **Choose which hours count** (see below).
4. **Run it.** Progress, an ETA and a cancel button appear in the footer. Each
   point is written to disk as it finishes, so an interrupted run resumes.
5. **Scrub the year.** The slider under the map steps through seasons or months.

## Which hours count

| Mode | Keeps | Notes |
|---|---|---|
| Astronomical night | Sun more than 18° below the horizon | True darkness. **None at all above ~48.5°N around midsummer.** |
| Sun below the horizon | Sunset to sunrise (centre below −0.833°) | Much more data at high latitude; only polar day leaves a gap. |
| Full 24 hours | Everything | Day and night alike. |

Solar position is computed per point, per hour, from the standard USNO/NOAA
algorithm — so the window follows latitude and time of year rather than a fixed
clock. Roughly what that costs you, over a year:

| | astronomical | sun below horizon |
|---|---|---|
| Tromsø 69.6°N | 1,861 h · 190 nights | 4,162 h · 296 nights |
| Trondheim 63.4°N | 2,059 h · 224 nights | 4,215 h · 365 nights |
| Oslo 59.9°N | 2,185 h · 243 nights | 4,241 h · 365 nights |
| Madrid 40.4°N | 3,072 h · 365 nights | 4,314 h · 365 nights |

Nights are indexed by the evening they begin and split at local solar noon, so a
span crossing midnight stays one night. No timezone setting is needed anywhere.

When a period has no qualifying hours anywhere in the area — a Scandinavian
summer under astronomical night — the map says so instead of drawing an empty
field.

## The time-of-year slider

Each stop is a **climatology**: every matching month pooled across your whole
date range. "Winter" means all Decembers, Januaries and Februaries together, not
winter 2025. Switch between four seasons and twelve months; the leftmost stop is
always the whole year.

## Export

*Map PNG* renders the cartopy figure the original script produced, honouring the
current period, colour ramp, banding and scale.

## How the data is cached

Cloud cover is fetched once per grid point per calendar year and reduced to one
mean per night:

| Path | What it holds |
|---|---|
| `.requests_http_cache.sqlite` | Raw archive responses, keyed by request URL |
| `.cache/<mode>/cache_<lat>_<lon>.json` | Nightly means per point |
| `.cache/_fast/<mode>/<lat>_<lon>.npz` | Binary mirror, so reloads skip JSON parsing |
| `.cache/_fast/<mode>/coverage.json` | Index of what each point covers, so the estimate is instant |

A cached file holds *nightly means*, not raw hours, so it cannot be re-cut for a
different definition of night. Each mode therefore gets its own namespace
(`astro`, `night`, `all24`) and switching between them refetches. The `_fast`
directory is derived — delete it any time and it rebuilds.

## Rate limits

Open-Meteo's free tier allows roughly 10,000 requests/day and 5,000/hour, refuses
bursts above about eight at once, and weights each request by how much data it
covers. *Parallel requests* defaults to 4. A run needs one request per point per
year, so a 60 × 60 grid over six years is 21,600 requests — more than a day's
allowance. The footer warns when an estimate exceeds it, and because progress is
saved per point you can simply run again the next day.

Measured throughput is about **2 point-years per second per connection**, and it
is data-volume-bound: batching many locations into one request does not help.

## When something goes wrong

Every run writes `.cache/logs/run-<id>.log`: the settings, the versions of every
library with native code in it, and one flushed line per archive request *before*
it is made — so if a run dies, the last line names what it was doing.

`faulthandler` is installed in every process and dumps all thread stacks to
`.cache/logs/fault-*.log` on a native crash. Downloading runs in separate worker
processes, so a crash there costs only the points in flight; the pool is rebuilt
and the run continues. Failures that are clearly not transient (a broken package
rather than a flaky network) are raised at once instead of being retried, and a
run where nothing at all succeeds aborts after eight points rather than grinding
through hundreds.

Logs are browsable at `/api/logs` and `/api/logs/<name>` while the server runs.

## The original script

`main.py` is untouched and still works. Edit the constants at the top and run
`python main.py` to write `map_output_overall.png` plus one PNG per season. Note
it uses its own fixed 18:00–06:00 window in a named timezone, and its own cache
layout, so it does not share the web app's namespaces.

## Layout

```
app.py                  Flask routes and the JSON API
cloudcover/
  config.py             Settings model, grid construction, cache paths
  solar.py              Solar position; astronomical night and sunset thresholds
  fetch.py              Archive access, per-point cache, pre-flight estimate
  analysis.py           Night aggregation, interpolation
  render.py             Static PNG export
  jobs.py               Background runs with progress, cancellation, crash recovery
  diagnostics.py        Run logs and native-crash dumps
templates/index.html    The page
static/css/app.css      Styles
static/js/app.js        Map, selection box, overlay rendering, controls
main.py                 The original standalone script
```

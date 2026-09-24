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

## Running your own archive (recommended)

The public tier is the slow part, not the data. Open-Meteo publish their whole
archive as open data plus a Docker image of the API server, and it serves the
*same* HTTP API — so the app only needs pointing at it.

```
python app.py
```

That is the whole thing. The app starts the archive container on launch, waits
for it to serve, and stops it again on exit. The startup banner says which
archive is in use, and `GET /api/archive` reports whether it is reachable.

| | |
|---|---|
| `python app.py` | manages the container for you |
| `python app.py --no-docker` | public tier instead; Docker is not touched |
| `python app.py --archive-url URL` | an archive you run elsewhere |
| `docker compose up -d` first | the app adopts it and leaves it running on exit |

Starting it yourself is worth doing if you want the chunk cache to stay warm
between app restarts — the app only stops what it started.

Shutdown is handled for Ctrl+C, a closed terminal and `taskkill`/SIGTERM. A
force-kill (`taskkill /F`) runs no handler at all and strands the container; the
next launch notices the orphan is one of ours and takes it down.

If Docker is missing or asleep the app says so and exits, rather than quietly
dropping you onto the rate-limited public tier.

Nothing is pre-downloaded. OM-files are chunked spatially *and* temporally —
roughly 3 x 3 grid cells by 120 timesteps, one to two kilobytes each — and the
server pulls individual chunks out of S3 with HTTP range requests as queries ask
for them, keeping what it fetched in `CACHE_SIZE` of local disk. Query Europe and
you pull Europe's chunks, not the globe.

(This is worth contrasting with ARCO-ERA5 on Google Cloud, which looks like the
obvious choice and is not: it is chunked `[1, 721, 1440]`, one whole global field
per hour, so extracting a 4 x 4 degree box over six years means pulling about
218 GB.)

Measured on a 25-point, 3-year job (4 x 4 degree box, 8 workers):

| | time | downloaded |
|---|---|---|
| Cold region, chunks fetched on demand | 42s | **37 MB** |
| Same region again, archive warm | **2.0s** | 0 MB |
| Public tier, for comparison | ~8s | n/a, but quota-limited |

A cold region is *slower* than the public API, because the container pulls from
S3 while open-meteo.com serves from its own warm storage. You win from the second
query onward, and never hit a quota.

### On-demand caching is per point, not per area

Points are fetched individually, so **raising the grid resolution over the same
box downloads cold again** — the finer grid lands between the chunks the coarser
one pulled. Measured on one 4 x 4 degree box:

| grid | points | pulled |
|---|---|---|
| 5 x 5 | 25 | 31 MB |
| 9 x 9, same box | 81 | +28 MB |
| 15 x 15, same box | 225 | +68 MB |

Prefetching just the box looks like the fix and is not: the archive serves ERA5
on its N320 Gaussian grid at about **0.0703 degrees** — successive cells come
back as 15.00879, 15.07909, 15.14938 — so a 2 x 2 degree box holds ~840 cells and
Europe ~355,000. Enumerating them costs far more than whole years, and any cell
missed is a cold spot the next finer grid finds.

### Syncing years is the fix

```
python app.py --sync-years 2019-2024
```

About **5.14 GB per year**. Once done, every point is local at any resolution,
for any region, and runs go at roughly 37 year-fetches/second. The footer shows
the exact command for the range you have selected.

With a self-hosted archive there is no quota: the daily-budget warning disappears
and the worker ceiling rises from 12 to 64.

## Rate limits on the public tier

Roughly 10,000 requests/day and 5,000/hour, bursts above about eight at once are
refused, and each request is weighted by how much data it covers. *Parallel
requests* defaults to 4. A run needs one request per point per year, so a 60 x 60
grid over six years is 21,600 requests — more than a day's allowance. The footer
warns when an estimate exceeds it, and because progress is saved per point you
can run again the next day.

Measured throughput is about **2 point-years per second per connection**, and it
is data-volume-bound: batching many locations into one request does not help.
Paid plans start at $29/month for 1M calls if you would rather not self-host.

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
docker-compose.yml      Self-hosted Open-Meteo archive
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

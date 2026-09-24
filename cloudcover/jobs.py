"""Background run management: progress, cancellation, logging, and a job registry.

Fetching happens in a pool of separate processes rather than threads. That is
partly what the original script did, and partly containment: a crash inside a
native extension takes down the process it happens in, and if that process is the
web server the whole run is lost with it. Isolated, the same crash costs only the
points in flight — the pool is rebuilt and the run carries on.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from . import analysis, diagnostics, fetch
from .config import Settings

MAX_JOBS_RETAINED = 6
MAX_FAILURES_REPORTED = 25
MAX_POOL_RESTARTS = 3
# How many consecutive point failures with zero successes before we call it.
SYSTEMIC_FAILURE_THRESHOLD = 8


class Job:
    def __init__(self, settings: Settings):
        self.id = uuid.uuid4().hex[:12]
        self.settings = settings
        self.status = "queued"  # queued | running | done | error | cancelled
        self.phase = "Starting"
        self.total = settings.point_count
        self.done = 0
        self.fetched = 0
        self.restarts = 0
        self.pool_failed = False
        self.systemic_error: str | None = None
        self.failures: list[str] = []
        self.error: str | None = None
        self.result: analysis.RunResult | None = None
        self.started = time.time()
        self.started_phase = self.started
        self.finished: float | None = None
        self.cancel_event = threading.Event()
        self.log_path = diagnostics.run_log_path(self.id)
        self._lock = threading.Lock()

    # ------------------------------------------------------------- progress

    def tick(self, fetched_years: int = 0) -> None:
        with self._lock:
            self.done += 1
            if fetched_years:
                self.fetched += fetched_years

    def begin_phase(self, phase: str, total: int) -> None:
        with self._lock:
            self.phase = phase
            self.done = 0
            self.total = max(total, 0)
            self.started_phase = time.time()

    def note_failure(self, message: str) -> None:
        with self._lock:
            if len(self.failures) < MAX_FAILURES_REPORTED:
                self.failures.append(message)

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()

    def snapshot(self) -> dict:
        with self._lock:
            done, total = self.done, self.total
            elapsed = (self.finished or time.time()) - self.started
            # ETA is paced off the current phase: reading cache and downloading
            # run at wildly different speeds, so blending them misleads.
            eta = None
            if self.status == "running" and done > 0 and done < total:
                phase_elapsed = time.time() - self.started_phase
                eta = (phase_elapsed / done) * (total - done)
            return {
                "id": self.id,
                "status": self.status,
                "phase": self.phase,
                "done": done,
                "total": total,
                "fetched_years": self.fetched,
                "restarts": self.restarts,
                "elapsed": round(elapsed, 1),
                "eta": round(eta, 1) if eta is not None else None,
                "failures": list(self.failures),
                "failure_count": len(self.failures),
                "error": self.error,
                "systemic_error": self.systemic_error,
                "has_result": self.result is not None,
                "log": os.path.basename(self.log_path),
                "settings": self.settings.to_dict(),
            }


_jobs: "OrderedDict[str, Job]" = OrderedDict()
_jobs_lock = threading.Lock()


def get(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def register(job: Job) -> None:
    with _jobs_lock:
        _jobs[job.id] = job
        while len(_jobs) > MAX_JOBS_RETAINED:
            _, old = _jobs.popitem(last=False)
            old.cancel()


# ------------------------------------------------------------------ the pool


def _drain(job: Job, points, per_point, pending: list[int], log) -> set[int]:
    """Run one pool over ``pending``; return the indices still outstanding.

    A ``BrokenProcessPool`` means a worker died without returning — the crash
    this whole arrangement exists to survive. Whatever finished first is kept.
    """
    remaining = set(pending)
    successes = 0
    settings = job.settings
    ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=settings.workers, mp_context=ctx)

    futures = {
        pool.submit(fetch.run_point, (settings, i, points[i][0], points[i][1], job.id)): i
        for i in pending
    }
    log.info("pool of %d workers started for %d points", settings.workers, len(pending))

    try:
        for future in as_completed(futures):
            if job.cancelled():
                log.info("cancellation requested; abandoning %d points", len(remaining))
                break
            index, days, means, fetched, failure = future.result()
            remaining.discard(index)
            if failure:
                job.note_failure(failure)
                # If nothing at all is succeeding, the fault is systemic —
                # a broken package, no network, a bad key. Grinding through
                # hundreds more points to learn the same thing wastes the
                # user's time, which is exactly what happened when
                # requests_cache broke.
                if successes == 0 and len(job.failures) >= SYSTEMIC_FAILURE_THRESHOLD:
                    job.systemic_error = failure
                    log.error("aborting: first %d points all failed, no successes. "
                              "Representative failure: %s",
                              len(job.failures), failure)
                    job.cancel_event.set()
                    break
            else:
                per_point[index] = (days, means)
                successes += 1
            job.tick(fetched)
            if job.done % 100 == 0:
                log.info("progress %d/%d (%d year-fetches)", job.done, job.total, job.fetched)
    except BrokenProcessPool as exc:
        log.error("WORKER PROCESS DIED: %s — %d points still outstanding. "
                  "Check the fault-worker-*.log files in this directory.",
                  exc, len(remaining))
        raise
    finally:
        try:
            pool.shutdown(wait=not job.cancelled(), cancel_futures=True)
        except Exception:
            pass

    return remaining


def _read_cache(job: Job, points, per_point, log) -> list[int]:
    """Load everything already on disk, in this process. Returns points still
    needing a download.

    Reading the cache never touches the network, so it is safe here and avoids
    paying process-spawn cost for a run that is already fully cached — which is
    the common case once an area has been surveyed.
    """
    job.begin_phase("Reading cached nights", len(points))
    to_fetch: list[int] = []
    lock = threading.Lock()

    def load(index):
        lat, lon = points[index]
        try:
            days, means, needs_fetch = fetch.cached_point(job.settings, lat, lon)
            if days.size:
                per_point[index] = (days, means)
            if needs_fetch:
                with lock:
                    to_fetch.append(index)
        except Exception as exc:
            log.warning("cache read failed for %.4f,%.4f: %s", lat, lon, exc)
            with lock:
                to_fetch.append(index)
        job.tick()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(load, range(len(points))))

    log.info("cache pass done: %d of %d points need downloading",
             len(to_fetch), len(points))
    return sorted(to_fetch)


def _download(job: Job, points, per_point, pending, log) -> None:
    """Fetch the outstanding points in worker processes, surviving crashes."""
    job.begin_phase("Downloading from the archive", len(pending))
    while pending and not job.cancelled():
        try:
            pending = sorted(_drain(job, points, per_point, pending, log))
            return
        except BrokenProcessPool:
            job.restarts += 1
            pending = sorted(i for i in pending if per_point[i][0] is None)
            if job.restarts > MAX_POOL_RESTARTS:
                job.pool_failed = True
                job.note_failure(
                    f"Worker processes crashed {job.restarts} times; giving up with "
                    f"{len(pending)} points unfinished. See {os.path.basename(job.log_path)}.")
                log.error("giving up after %d pool restarts", job.restarts)
                return
            job.phase = f"Recovering from a worker crash ({job.restarts})"
            log.warning("restarting pool (%d/%d), %d points left",
                        job.restarts, MAX_POOL_RESTARTS, len(pending))
            time.sleep(2.0)


def _run(job: Job) -> None:
    settings = job.settings
    points = settings.grid_points()
    per_point = [(None, None)] * len(points)

    log = diagnostics.run_logger(job.id)
    log.info("=" * 72)
    log.info("run %s starting: %d points, %s to %s, night %s",
             job.id, len(points), settings.start_date, settings.end_date,
             settings.night_label())
    log.info("settings: %s", json.dumps(settings.to_dict(), sort_keys=True))
    log.info("environment:\n%s", diagnostics.environment_report())

    job.status = "running"
    settings.ensure_cache_dirs()

    try:
        pending = _read_cache(job, points, per_point, log)
        if pending and not job.cancelled():
            _download(job, points, per_point, pending, log)

        # Index what landed on disk, even on a cancelled or crashed run — it is
        # what keeps the next pre-flight estimate instant instead of thousands
        # of reads, and it makes a re-run resume cheaply.
        fetch.write_coverage(settings, {
            fetch.point_key(lat, lon): [int(days.min()), int(days.max()), int(days.size)]
            for (lat, lon), (days, _) in zip(points, per_point)
            if days is not None and days.size
        })

        if job.systemic_error:
            job.status = "error"
            job.phase = "Failed"
            job.error = (
                f"Every point failed the same way, so the run was stopped early rather "
                f"than retrying hundreds more. First failure: {job.systemic_error}"
            )
            log.error(job.error)
            return

        if job.cancelled():
            job.status = "cancelled"
            job.phase = "Cancelled"
            log.info("run cancelled after %d/%d points", job.done, job.total)
            return

        job.phase = "Aggregating nights"
        job.result = analysis.build_result(settings, points, per_point, job.failures)
        log.info("aggregated %d nights across %d points",
                 job.result.n_nights, job.result.n_points)

        if job.result.n_nights == 0:
            job.status = "error"
            # Blaming the date range when the workers never started would send
            # anyone debugging in exactly the wrong direction.
            if job.pool_failed:
                job.error = (
                    f"The download workers could not run — every one of {job.restarts} "
                    f"attempts died on startup. See .cache/logs/{os.path.basename(job.log_path)}. "
                    "If you launched the server from your own script rather than "
                    "`python app.py`, that script needs an "
                    "`if __name__ == \"__main__\":` guard, which Windows requires "
                    "before any process pool can start.")
            else:
                job.error = (
                    "No nightly data came back for this area and date range. "
                    "Check the dates and that the region is covered by the archive.")
            job.phase = "Failed"
            log.error(job.error)
            return

        if job.pool_failed:
            job.note_failure(
                "Some points are missing because the download workers kept crashing; "
                "re-running will retry only those.")

        job.status = "done"
        job.phase = "Complete"
        log.info("run complete in %.1fs (%d failures, %d pool restarts)",
                 time.time() - job.started, len(job.failures), job.restarts)
    except Exception as exc:
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.phase = "Failed"
        log.exception("run failed")
        traceback.print_exc()
    finally:
        job.finished = time.time()
        for handler in log.handlers:
            try:
                handler.flush()
            except Exception:
                pass


def start(settings: Settings) -> Job:
    job = Job(settings)
    register(job)
    thread = threading.Thread(target=_run, args=(job,), name=f"run-{job.id}", daemon=True)
    thread.start()
    return job

"""Lifecycle for the self-hosted Open-Meteo archive container.

The app brings the container up on start and takes it down on exit, so there is
one thing to run rather than two. Two rules keep that from being annoying:

*   if the container was already up when we arrived, we leave it up — you may be
    running it deliberately, and pulling it out from under yourself on quit
    would be rude;
*   a Docker that is missing or asleep is reported plainly, with the flags that
    skip it, rather than silently dropping to the rate-limited public tier.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPOSE_FILE = os.path.join(PROJECT_ROOT, "docker-compose.yml")
CONTAINER_NAME = "open-meteo"
LOCAL_ARCHIVE_URL = "http://127.0.0.1:8080/v1/archive"

# Written while we own the container, removed when we stop it. A force-killed
# app (taskkill /F, SIGKILL) runs no handler and strands the container; finding
# this marker next time tells us the orphan is ours to take back and shut down,
# rather than one the user started deliberately.
OWNER_MARKER = os.path.join(PROJECT_ROOT, ".cache", ".archive-owned")

# Pulling the image on a cold machine is the slow part, not the boot.
START_TIMEOUT = 300.0
STOP_TIMEOUT = 60.0


class ContainerError(RuntimeError):
    """Docker is unavailable, or the archive would not come up."""


def _docker(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def docker_available() -> tuple[bool, str]:
    """Is the CLI installed and the daemon accepting connections?"""
    try:
        probe = _docker("info", "--format", "{{.ServerVersion}}", timeout=30)
    except FileNotFoundError:
        return False, "the docker command was not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "docker did not respond within 30s"
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip().splitlines()
        hint = detail[-1] if detail else "docker info failed"
        return False, f"the Docker daemon is not reachable ({hint})"
    return True, probe.stdout.strip()


def _claim() -> None:
    try:
        os.makedirs(os.path.dirname(OWNER_MARKER), exist_ok=True)
        with open(OWNER_MARKER, "w") as fh:
            fh.write(str(os.getpid()))
    except OSError:
        pass


def _release() -> None:
    try:
        os.remove(OWNER_MARKER)
    except OSError:
        pass


def _claimed() -> bool:
    return os.path.exists(OWNER_MARKER)


def is_running() -> bool:
    try:
        seen = _docker("ps", "--filter", f"name=^{CONTAINER_NAME}$",
                       "--format", "{{.Names}}", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return seen.returncode == 0 and CONTAINER_NAME in seen.stdout.split()


def archive_responds(url: str = LOCAL_ARCHIVE_URL, timeout: float = 5.0) -> bool:
    """One cheap real query — 'container up' is not the same as 'serving'."""
    probe = (f"{url}?latitude=52&longitude=5&start_date=2023-01-01"
             f"&end_date=2023-01-01&hourly=cloud_cover")
    try:
        with urllib.request.urlopen(probe, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def start(log=print) -> bool:
    """Bring the archive up and wait for it to serve.

    Returns True if this call started it — the caller should then stop it on the
    way out. False means it was already running and is not ours to touch.
    """
    ok, detail = docker_available()
    if not ok:
        raise ContainerError(
            f"Cannot start the local archive: {detail}.\n"
            f"  Start Docker Desktop and try again, or run with --no-docker to use\n"
            f"  the public Open-Meteo tier, or --archive-url to point somewhere else."
        )

    if is_running() and archive_responds():
        if _claimed():
            # Left behind by an app that was killed outright. Ours again.
            log(f"  Archive               ->  reclaiming container left by a previous run")
            return True
        log(f"  Archive               ->  already running at {LOCAL_ARCHIVE_URL}")
        return False

    if not os.path.exists(COMPOSE_FILE):
        raise ContainerError(f"No docker-compose.yml at {COMPOSE_FILE}")

    log("  Archive               ->  starting container (first run pulls the image)...")
    up = _docker("compose", "up", "-d", timeout=START_TIMEOUT)
    if up.returncode != 0:
        raise ContainerError(
            "docker compose up failed:\n"
            + (up.stderr or up.stdout or "").strip()
        )

    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if archive_responds():
            _claim()
            return True
        if not is_running():
            logs = _docker("compose", "logs", "--tail", "20", timeout=30)
            raise ContainerError(
                "The archive container exited while starting:\n"
                + (logs.stdout or logs.stderr or "").strip()
            )
        time.sleep(1.0)

    raise ContainerError(
        f"The archive container did not answer within {START_TIMEOUT:.0f}s. "
        f"Check `docker compose logs`."
    )


def stop(log=print) -> None:
    """Take the container down, leaving the cache volume in place."""
    _release()
    try:
        log("  stopping the archive container...")
        down = _docker("compose", "down", timeout=STOP_TIMEOUT)
        if down.returncode != 0:
            log("  (could not stop it: "
                + (down.stderr or down.stdout or "").strip().splitlines()[-1] + ")")
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"  (could not stop it: {type(exc).__name__}: {exc})")


def sync_years(year_range: str, log=print) -> int:
    """Download whole years of cloud cover into the container's volume.

    The alternative - warming only the selected box - cannot work: the archive
    sits on ERA5's N320 Gaussian grid at ~0.0703 degrees, so a box holds far too
    many cells to enumerate, and any cell missed is a cold spot a finer grid
    will find. Whole years are simpler, cheaper past a couple of degrees square,
    and leave every point everywhere local.

    About 5.14 GB per year. Streams progress straight through.
    """
    ok, detail = docker_available()
    if not ok:
        raise ContainerError(f"Cannot sync: {detail}")

    log(f"  syncing copernicus_era5 cloud_cover for {year_range}")
    log("  (about 5.14 GB per year; this runs once and then every run is local)")
    proc = subprocess.Popen(
        ["docker", "compose", "run", "--rm", "openmeteo", "sync",
         "copernicus_era5", "cloud_cover", "--year", year_range],
        cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        log("    " + line.rstrip())
    return proc.wait()

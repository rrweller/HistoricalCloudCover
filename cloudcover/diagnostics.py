"""Crash and run logging.

The failure this exists for is a *native* crash — the process dies with no Python
traceback, so ordinary logging only records that work stopped, never where. Two
things close that gap:

*   ``faulthandler``, which dumps the Python stack of every thread when the
    interpreter takes a SIGSEGV or a Windows access violation;
*   a flushed-per-record run log whose last line is, by construction, the last
    thing the process was doing.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import platform
import sys
import threading

LOG_DIR = os.path.join(".cache", "logs")
_fault_file = None
_fault_lock = threading.Lock()
_configured: set[str] = set()


def log_dir() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR


def enable_faulthandler(tag: str = "main") -> str | None:
    """Install the fault handler for this process. Safe to call repeatedly.

    The stream has to stay open for the life of the process, so the handle is
    parked in a module global rather than being closed.
    """
    global _fault_file
    with _fault_lock:
        if _fault_file is not None:
            return None
        try:
            path = os.path.join(log_dir(), f"fault-{tag}-{os.getpid()}.log")
            _fault_file = open(path, "a", buffering=1)
            _fault_file.write(f"\n=== {tag} pid={os.getpid()} started ===\n")
            faulthandler.enable(file=_fault_file, all_threads=True)
            return path
        except Exception:
            return None


def run_log_path(run_id: str) -> str:
    return os.path.join(log_dir(), f"run-{run_id}.log")


def run_logger(run_id: str) -> logging.Logger:
    """A logger writing to this run's own file, flushed on every record.

    Worker processes attach to the same file; each record carries its pid so an
    interleaved timeline still reads cleanly.
    """
    name = f"cloudcover.run.{run_id}"
    logger = logging.getLogger(name)
    if name in _configured:
        return logger

    with _fault_lock:
        if name in _configured:
            return logger
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        try:
            handler = logging.FileHandler(run_log_path(run_id), mode="a", encoding="utf-8")
            handler.setFormatter(logging.Formatter(
                "%(asctime)s pid=%(process)-6d %(threadName)-12s %(levelname)-7s %(message)s",
                datefmt="%H:%M:%S",
            ))
            logger.addHandler(handler)
        except Exception:
            logger.addHandler(logging.NullHandler())
        _configured.add(name)
    return logger


def environment_report() -> str:
    """Versions of everything with native code in it, for the top of a run log."""
    lines = [
        f"python   {sys.version.split()[0]} ({platform.machine()})",
        f"platform {platform.platform()}",
    ]
    from importlib import metadata

    for mod, dist in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
                      ("requests", "requests"), ("urllib3", "urllib3"),
                      ("certifi", "certifi"), ("requests_cache", "requests-cache"),
                      ("openmeteo_requests", "openmeteo-requests"),
                      ("openmeteo_sdk", "openmeteo-sdk"),
                      ("flatbuffers", "flatbuffers"),
                      ("retry_requests", "retry-requests")):
        version = None
        try:
            version = getattr(__import__(mod), "__version__", None)
        except Exception:
            pass
        if not version:
            try:
                version = metadata.version(dist)
            except Exception:
                version = "not installed"
        lines.append(f"{mod:<20} {version}")
    try:
        import ssl

        lines.append(f"{'openssl':<20} {ssl.OPENSSL_VERSION}")
    except Exception:
        pass
    return "\n".join("    " + line for line in lines)


def recent_logs(limit: int = 12) -> list[dict]:
    """Newest-first listing of what is in the log directory."""
    try:
        entries = []
        for name in os.listdir(LOG_DIR):
            path = os.path.join(LOG_DIR, name)
            if os.path.isfile(path):
                entries.append({
                    "name": name,
                    "size": os.path.getsize(path),
                    "modified": os.path.getmtime(path),
                })
        entries.sort(key=lambda e: e["modified"], reverse=True)
        return entries[:limit]
    except Exception:
        return []

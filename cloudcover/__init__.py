"""Historical nighttime cloud-cover analysis over a geographic grid.

The package is a refactor of the original ``main.py`` script into reusable
pieces so both the CLI and the web UI can drive the same pipeline:

    config    settings model, grid construction, cache namespacing
    fetch     Open-Meteo archive access + per-point on-disk cache
    analysis  night aggregation, metrics, interpolation, rankings
    render    matplotlib/cartopy static map export
    jobs      background job runner with progress + cancellation
"""

__all__ = ["config", "fetch", "analysis", "render", "jobs"]

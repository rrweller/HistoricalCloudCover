"""Static map export — the original cartopy figure, parameterised.

Kept deliberately close to the script's output so exported PNGs still look like
the ones already in the repo; the differences are that bounds, colormap, banding
and labels now come from the caller.
"""

from __future__ import annotations

import io
import threading

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

# matplotlib's pyplot state machine is not thread-safe; exports are rare and
# cheap enough to simply serialise.
_render_lock = threading.Lock()

# Sequential ramp shared with the browser overlay, so an export matches the map.
_BLUE = [
    "#eaf2fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
    "#104281", "#0d366b",
]

_CMAPS = {
    "blue": LinearSegmentedColormap.from_list("cc_blue", _BLUE),
    "viridis": "viridis",
    "magma": "magma",
    "ylgnbu": "YlGnBu",
    "greys": "Greys",
}


def _cmap(name: str):
    return _CMAPS.get((name or "blue").lower(), _CMAPS["blue"])


def _fig_size(lat_min, lat_max, lon_min, lon_max, base_height=11.0):
    lat_span = max(lat_max - lat_min, 1e-6)
    lon_span = max(lon_max - lon_min, 1e-6)
    width = base_height * (lon_span / lat_span)
    return (max(5.0, min(26.0, width)), base_height)


def export_png(
    lats,
    lons,
    values,
    bounds,
    *,
    title: str,
    subtitle: str = "",
    legend_label: str = "Cloud cover (%)",
    colormap: str = "blue",
    vmin: float = 0.0,
    vmax: float = 100.0,
    bands: int = 10,
    show_points: bool = True,
    resolution: int = 700,
    method: str = "cubic",
) -> bytes:
    """Render the interpolated field to PNG bytes."""
    lat_min, lat_max, lon_min, lon_max = bounds

    with _render_lock:
        lat_span = max(lat_max - lat_min, 1e-6)
        lon_span = max(lon_max - lon_min, 1e-6)
        height = int(max(120, min(1400, resolution)))
        width = int(max(120, min(1800, height * lon_span / lat_span)))

        # Even latitude spacing here: cartopy reprojects the mesh itself.
        grid_lats = np.linspace(lat_min, lat_max, height)
        grid_lons = np.linspace(lon_min, lon_max, width)
        lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)

        ok = np.isfinite(values)
        if ok.sum() < 4:
            raise ValueError("Not enough valid points to render a map")
        pts = np.column_stack([np.asarray(lons)[ok], np.asarray(lats)[ok]])
        field = griddata(pts, np.asarray(values)[ok], (lon_mesh, lat_mesh), method=method)
        holes = ~np.isfinite(field)
        if holes.any():
            field[holes] = griddata(
                pts, np.asarray(values)[ok], (lon_mesh[holes], lat_mesh[holes]), method="nearest"
            )

        center_lat = (lat_min + lat_max) / 2.0
        center_lon = (lon_min + lon_max) / 2.0
        proj = ccrs.AlbersEqualArea(central_latitude=center_lat, central_longitude=center_lon)

        fig = plt.figure(figsize=_fig_size(lat_min, lat_max, lon_min, lon_max))
        ax = plt.axes(projection=proj)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        states = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="50m",
            facecolor="none",
        )
        # cfeature.LAND and cfeature.OCEAN cost ~40 s between them here: their
        # polygons are global and reprojecting them into Albers at draw time
        # dominates the render. They also sit *under* a contourf that covers the
        # whole extent, so nothing of them is ever visible. A flat backdrop for
        # any uncovered sliver gives the identical picture in about two seconds.
        ax.patch.set_facecolor("#e9e6dd")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAKES, alpha=0.8)
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(states, edgecolor="gray")

        levels = np.linspace(vmin, vmax, max(2, int(bands)) + 1)
        low = bool(np.nanmin(field) < vmin)
        high = bool(np.nanmax(field) > vmax)
        extend = "both" if (low and high) else "min" if low else "max" if high else "neither"
        contour = ax.contourf(
            lon_mesh,
            lat_mesh,
            field,
            levels=levels,
            cmap=_cmap(colormap),
            vmin=vmin,
            vmax=vmax,
            extend=extend,
            transform=ccrs.PlateCarree(),
        )
        cbar = plt.colorbar(contour, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_label(legend_label)

        if show_points:
            ax.scatter(
                np.asarray(lons), np.asarray(lats),
                c="black", s=1, transform=ccrs.PlateCarree(),
            )

        ax.set_title(f"{title}\n{subtitle}" if subtitle else title)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

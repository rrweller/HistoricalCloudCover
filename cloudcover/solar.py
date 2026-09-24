"""Solar position, for deciding when a place is actually in astronomical night.

Astronomical night is the span when the sun sits more than 18° below the
horizon — the point past which sunlight no longer contributes to sky glow. How
long that lasts, and whether it happens at all, depends on latitude and the time
of year: above roughly 48.5° it disappears entirely around midsummer, and inside
the polar circles it can be absent for months.

This is the standard low-precision solar position algorithm (USNO / NOAA). It is
good to about a minute of arc, which is far finer than an 18° threshold sampled
on hourly data needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sun centre this far below the horizon: no astronomical twilight remains.
ASTRONOMICAL_DEPRESSION = -18.0

# Sunset/sunrise: the centre sits slightly below the horizon when the upper limb
# disappears, once refraction (~34') and the solar semidiameter (~16') are taken
# off. This is the ordinary "it is dark out" threshold, and at high latitudes it
# yields far more usable nights than the astronomical one.
HORIZON_DEPRESSION = -0.833

J2000 = 2451545.0


def solar_altitude(times_utc: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Altitude of the sun's centre in degrees, for each timestamp.

    Negative means below the horizon. Geometric altitude — no refraction term,
    which matters near the horizon but not 18° down.
    """
    days = np.asarray(times_utc.to_julian_date(), dtype=np.float64) - J2000

    # Mean longitude and mean anomaly of the sun, degrees.
    mean_long = np.radians((280.460 + 0.9856474 * days) % 360.0)
    mean_anom = np.radians((357.528 + 0.9856003 * days) % 360.0)

    # Ecliptic longitude, applying the equation of centre.
    ecliptic = mean_long + np.radians(
        1.915 * np.sin(mean_anom) + 0.020 * np.sin(2.0 * mean_anom)
    )

    obliquity = np.radians(23.439 - 0.0000004 * days)

    declination = np.arcsin(np.sin(obliquity) * np.sin(ecliptic))
    right_asc = np.arctan2(np.cos(obliquity) * np.sin(ecliptic), np.cos(ecliptic))

    # Greenwich mean sidereal time -> local hour angle.
    gmst_hours = (18.697374558 + 24.06570982441908 * days) % 24.0
    local_sidereal = np.radians((gmst_hours * 15.0 + lon) % 360.0)
    hour_angle = local_sidereal - right_asc

    lat_r = np.radians(lat)
    sin_alt = (
        np.sin(lat_r) * np.sin(declination)
        + np.cos(lat_r) * np.cos(declination) * np.cos(hour_angle)
    )
    return np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))


def sun_below(times_utc: pd.DatetimeIndex, lat: float, lon: float,
              depression: float = ASTRONOMICAL_DEPRESSION) -> np.ndarray:
    """Boolean mask of timestamps where the sun sits below ``depression`` degrees."""
    return solar_altitude(times_utc, lat, lon) < depression

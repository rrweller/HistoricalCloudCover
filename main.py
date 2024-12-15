import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import json
from datetime import timedelta, datetime
import pytz
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
from tqdm import tqdm

###############################################
# CONFIGURATION
###############################################
logging_enabled = False
log_file = "debug.log"

start_date = "2012-01-01"
end_date = "2024-11-30"

latitude_bounds = (57.5, 70)
longitude_bounds = (4.5, 15)
grid_resolution = 20
num_interp_points = grid_resolution * grid_resolution
output_json = "binned_nighttime_clouds.json"
tz = pytz.timezone("Europe/Oslo")

if logging_enabled:
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
else:
    logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

logging.info("Program started.")

###############################################
# HELPER FUNCTIONS
###############################################

def ensure_cache_dir():
    if not os.path.exists('.cache'):
        os.makedirs('.cache')

def compute_fig_size():
    lat_diff = latitude_bounds[1] - latitude_bounds[0]
    lon_diff = longitude_bounds[1] - longitude_bounds[0]
    aspect_ratio = lon_diff / lat_diff
    base_height = 10
    width = base_height * aspect_ratio
    height = base_height
    return (width, height)

def get_night_start_date(row):
    h = row["hour_local"]
    d = row["date_local"]
    if h >= 18:
        return d
    elif h < 6:
        return d - timedelta(days=1)
    else:
        return None

def all_dates_in_range(start_date, end_date):
    """Return a list of all dates between start_date and end_date inclusive."""
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    all_dates = []
    current = start_dt
    while current <= end_dt:
        all_dates.append(current.date())
        current += timedelta(days=1)
    return all_dates

def yearly_intervals(start_date, end_date):
    """Generate (year_start, year_end) tuples for each year in the range."""
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    intervals = []
    current_year = start_dt.year
    while current_year <= end_dt.year:
        year_start = datetime(current_year, 1, 1)
        year_end = datetime(current_year, 12, 31)
        if year_start < start_dt:
            year_start = start_dt
        if year_end > end_dt:
            year_end = end_dt
        intervals.append((year_start.date().isoformat(), year_end.date().isoformat()))
        current_year += 1
    return intervals

def load_cache(lat, lon):
    cache_file = f".cache/cache_{lat}_{lon}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}

def save_cache(lat, lon, data):
    cache_file = f".cache/cache_{lat}_{lon}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)

def fetch_yearly_data(lat, lon, start_yr, end_yr, session):
    """Fetch data for one entire year range."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_yr,
        "end_date": end_yr,
        "hourly": "cloud_cover"
    }

    while True:
        try:
            responses = openmeteo_requests.Client(session=session).weather_api(url, params=params)
            response = responses[0]
            break
        except Exception:
            time.sleep(60)
            continue

    hourly = response.Hourly()
    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    cloud_cover = hourly.Variables(0).ValuesAsNumpy()

    df = pd.DataFrame({
        "time_utc": times,
        "cloud_cover": cloud_cover
    })

    # Convert UTC to local time
    df["time_local"] = df["time_utc"].dt.tz_convert(tz)
    df["hour_local"] = df["time_local"].dt.hour
    df["date_local"] = df["time_local"].dt.date

    df["night_start_date"] = df.apply(get_night_start_date, axis=1)
    df_night = df[df["night_start_date"].notnull()]

    yearly_averages = {}
    for night_date, group in df_night.groupby("night_start_date"):
        night_str = night_date.isoformat()
        avg_cc = group["cloud_cover"].mean() if len(group) > 0 else np.nan
        yearly_averages[night_str] = {"average": float(avg_cc)}

    return yearly_averages

def process_location(lat, lon, start_date, end_date):
    ensure_cache_dir()
    cache_session = requests_cache.CachedSession('.requests_http_cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

    # Load existing data
    cached_data = load_cache(lat, lon)

    all_dates_list = all_dates_in_range(start_date, end_date)
    existing_nights = set(cached_data.keys())
    # Filter to only valid ISO date keys
    existing_nights = {d for d in existing_nights if is_iso_date(d)}

    all_dates_str = {d.isoformat() for d in all_dates_list}
    missing_nights = all_dates_str - existing_nights

    # If missing_nights is empty, we already have all data
    # If not, we need to fetch data for those years
    # Group missing nights by year
    while missing_nights:
        years_needed = {}
        for mn in missing_nights:
            y = datetime.fromisoformat(mn).year
            if y not in years_needed:
                years_needed[y] = []
            years_needed[y].append(mn)

        # For each year needed, fetch that entire year
        year_intervals = yearly_intervals(start_date, end_date)
        # year_intervals gives us start/end for each year in the range
        # We only fetch years that have missing nights
        for (ystart, yend) in year_intervals:
            y = datetime.fromisoformat(ystart).year
            if y in years_needed:
                # Fetch this year
                yearly_data = fetch_yearly_data(lat, lon, ystart, yend, retry_session)
                # Merge into cache
                for k, v in yearly_data.items():
                    cached_data[k] = v
                save_cache(lat, lon, cached_data)

        # Recalculate missing_nights after fetching
        cached_data = load_cache(lat, lon)
        existing_nights = set(d for d in cached_data.keys() if is_iso_date(d))
        missing_nights = all_dates_str - existing_nights

        # If after fetching all needed years some nights still missing, they might just not exist in the data.
        # Usually should not happen. We will break out if no improvement is possible.
        # But let's trust the API that we got all data now.

        # If the API doesn't provide all nights, missing_nights might remain.
        # We'll just continue with what we have.

        break  # We did one pass of fetching needed years. In normal conditions, this should fill everything.

    return (lat, lon, cached_data)

def is_iso_date(s):
    # Check if s is a valid ISO date (YYYY-MM-DD)
    try:
        datetime.fromisoformat(s)
        return True
    except ValueError:
        return False

def process_args(args):
    return process_location(*args)

def determine_season(date):
    m = date.month
    if (m == 12) or (m == 1) or (m == 2):
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def compute_interpolation(locs, vals, latitude_bounds, longitude_bounds, num_interp_points):
    lat_grid_interp = np.linspace(latitude_bounds[0], latitude_bounds[1], num_interp_points)
    lon_grid_interp = np.linspace(longitude_bounds[0], longitude_bounds[1], num_interp_points)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid_interp, lat_grid_interp)

    interpolated_clouds = None
    if len(vals) > 0 and not np.all(np.isnan(vals)):
        try:
            interpolated_clouds = griddata(
                (locs[:,0], locs[:,1]),
                vals,
                (lon_mesh, lat_mesh),
                method='cubic'
            )
        except Exception as e:
            logging.error(f"Interpolation error: {e}", exc_info=True)
            interpolated_clouds = None
    return lon_mesh, lat_mesh, interpolated_clouds

def plot_map(locs, lon_mesh, lat_mesh, interpolated_clouds, title, outfile):
    center_lat = (latitude_bounds[0] + latitude_bounds[1]) / 2
    center_lon = (longitude_bounds[0] + longitude_bounds[1]) / 2
    proj = ccrs.AlbersEqualArea(central_latitude=center_lat, central_longitude=center_lon)
    fig = plt.figure(figsize=compute_fig_size())
    ax = plt.axes(projection=proj)
    
    ax.set_extent([longitude_bounds[0], longitude_bounds[1], latitude_bounds[0], latitude_bounds[1]], crs=ccrs.PlateCarree())
    ax.set_title(f"Center: ({center_lat:.2f}, {center_lon:.2f})")

    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(states_provinces, edgecolor='gray')

    if interpolated_clouds is not None:
        levels = np.linspace(20, 100, 9)
        contour = ax.contourf(lon_mesh, lat_mesh, interpolated_clouds,
            levels=levels,
            cmap='YlGnBu',
            vmin=20,
            vmax=100,
            transform=ccrs.PlateCarree())
        cbar = plt.colorbar(contour, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Average Nighttime Cloud Cover (%)')
        ax.scatter(locs[:,0], locs[:,1], c='black',
                   transform=ccrs.PlateCarree(), s=4)
        plt.title(title)
        plt.savefig(outfile)
        logging.info(f"Map plotted and saved to {outfile}")
    else:
        logging.warning("No interpolated data available. Skipping map plotting.")

    plt.close(fig)

def main():
    buffer = 1.0
    extended_lat_bounds = (latitude_bounds[0] - buffer, latitude_bounds[1] + buffer)
    extended_lon_bounds = (longitude_bounds[0] - buffer, longitude_bounds[1] + buffer)

    lats = np.linspace(extended_lat_bounds[0], extended_lat_bounds[1], grid_resolution)
    lons = np.linspace(extended_lon_bounds[0], extended_lon_bounds[1], grid_resolution)

    tasks = [(lat, lon, start_date, end_date) for lat in lats for lon in lons]
    total_points = len(tasks)
    logging.info(f"Number of points to process: {total_points}")
    logging.info(f"Date range: {start_date} to {end_date}")

    binned_data = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_args, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=total_points, desc="Processing grid points"):
            lat, lon, nightly_averages = f.result()
            loc_key = f"{lat}_{lon}"
            binned_data[loc_key] = nightly_averages

    logging.info("All points processed. Converting to DataFrame and computing seasonal averages.")

    rows = []
    for loc_key, nights in binned_data.items():
        lat_str, lon_str = loc_key.split("_")
        lat_val = float(lat_str)
        lon_val = float(lon_str)

        # Filter only valid ISO date keys and valid data
        for night_str, vals in nights.items():
            if is_iso_date(night_str) and isinstance(vals, dict) and "average" in vals:
                avg_val = vals["average"]
                date_val = datetime.fromisoformat(night_str).date()
                rows.append((lat_val, lon_val, date_val, avg_val))

    df_all = pd.DataFrame(rows, columns=["lat", "lon", "date", "average"])
    df_all["season"] = df_all["date"].apply(determine_season)

    # Compute overall averages
    df_overall = df_all.groupby(["lat", "lon"])["average"].mean().reset_index()

    # Compute seasonal averages
    df_seasonal = df_all.groupby(["lat", "lon", "season"])["average"].mean().reset_index()

    if logging_enabled == True:
        try:
            with open(output_json, "w") as f:
                json.dump(binned_data, f, indent=2, default=str)
            logging.info(f"Saved nightly averages to {output_json}")
        except Exception as e:
            logging.error(f"Error saving JSON file: {e}", exc_info=True)

    # Interpolate and plot overall
    locs_overall = df_overall[["lon", "lat"]].values
    vals_overall = df_overall["average"].values
    lon_mesh, lat_mesh, interpolated_clouds_overall = compute_interpolation(locs_overall, vals_overall, extended_lat_bounds, extended_lon_bounds, num_interp_points)
    plot_map(locs_overall, lon_mesh, lat_mesh, interpolated_clouds_overall,
             f"Average Nighttime Cloud Cover in Norway\n({start_date} to {end_date})",
             "map_output_overall.png")

    # Seasons
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    for season in seasons:
        df_seas = df_seasonal[df_seasonal["season"] == season]
        locs_seas = df_seas[["lon", "lat"]].values
        vals_seas = df_seas["average"].values
        lon_mesh, lat_mesh, interpolated_clouds_seas = compute_interpolation(locs_seas, vals_seas, extended_lat_bounds, extended_lon_bounds, num_interp_points)
        plot_map(locs_seas, lon_mesh, lat_mesh, interpolated_clouds_seas,
                 f"Average Nighttime Cloud Cover in Norway ({season})\n({start_date} to {end_date})",
                 f"map_output_{season.lower()}.png")

    logging.info("Program finished successfully.")

if __name__ == "__main__":
    main()
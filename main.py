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

# Last 10 years of data:
start_date = "2023-01-01"
end_date = "2023-12-31"

latitude_bounds = (57.5, 70)
longitude_bounds = (4.5, 15)
grid_resolution = 4
num_interp_points = grid_resolution*grid_resolution
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

    # Assuming a base height of 10 units
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

def determine_missing_dates(lat, lon, start_date, end_date):
    """Check the cache for existing data and determine which daily ranges are missing."""
    cache_file = f".cache/cache_{lat}_{lon}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
    else:
        cached_data = {}

    # cached_data: { "YYYY-MM-DD": {"average": val}, ... }

    # Generate all dates in the requested range
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    all_dates = []
    current = start_dt
    while current <= end_dt:
        all_dates.append(current.date())
        current += timedelta(days=1)

    missing_dates = [d for d in all_dates if d.isoformat() not in cached_data]
    return missing_dates, cached_data

def fetch_daily_data(lat, lon, date_str, session):
    """Fetch data from the API for a single day."""
    openmeteo = openmeteo_requests.Client(session=session)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "cloud_cover"
    }

    while True:
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            break
        except Exception:
            time.sleep(62)
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

    # Compute nightly averages for this date (should be just one night_start_date)
    nightly_averages = {}
    for night_date, group in df_night.groupby("night_start_date"):
        night_str = night_date.isoformat()
        avg_cc = group["cloud_cover"].mean() if len(group) > 0 else np.nan
        nightly_averages[night_str] = {"average": float(avg_cc)}

    return nightly_averages

def update_cache(lat, lon, new_data):
    """Merge new data into the cache file for the given lat/lon."""
    cache_file = f".cache/cache_{lat}_{lon}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
    else:
        cached_data = {}

    for night_str, vals in new_data.items():
        cached_data[night_str] = vals

    with open(cache_file, "w") as f:
        json.dump(cached_data, f, indent=2)

def process_location(lat, lon, start_date, end_date):
    ensure_cache_dir()

    # Setup a cached session for requests
    cache_session = requests_cache.CachedSession('.requests_http_cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

    # Determine which dates are missing from the cache
    missing_dates, cached_data = determine_missing_dates(lat, lon, start_date, end_date)

    # Fetch missing data day-by-day
    for d in missing_dates:
        d_str = d.isoformat()
        daily_data = fetch_daily_data(lat, lon, d_str, retry_session)
        # Merge into cache
        update_cache(lat, lon, daily_data)

    # Reload the full cache after updates
    cache_file = f".cache/cache_{lat}_{lon}.json"
    with open(cache_file, "r") as f:
        final_cached_data = json.load(f)

    nightly_averages = final_cached_data
    return (lat, lon, nightly_averages)

def process_args(args):
    return process_location(*args)

def determine_season(date):
    # date is a datetime.date
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
        levels = np.linspace(0, 100, 11)
        contour = ax.contourf(lon_mesh, lat_mesh, interpolated_clouds,
                              levels=levels,
                              cmap='YlGnBu',
                              vmin=0,
                              vmax=100,
                              transform=ccrs.PlateCarree())
        cbar = plt.colorbar(contour, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Average Nighttime Cloud Cover (%)')
        ax.scatter(locs[:,0], locs[:,1], c='white', edgecolor='black',
                   transform=ccrs.PlateCarree(), s=4)
        plt.title(title)
        plt.savefig(outfile)
        logging.info(f"Map plotted and saved to {outfile}")
    else:
        logging.warning("No interpolated data available. Skipping map plotting.")

    plt.close(fig)

def main():
    # Define the grid of points with additional buffer points
    buffer = 1 # Add points 1 degree outside the bounds
    
    # Extend the bounds for the grid
    extended_lat_bounds = (latitude_bounds[0] - buffer, latitude_bounds[1] + buffer)
    extended_lon_bounds = (longitude_bounds[0] - buffer, longitude_bounds[1] + buffer)
    
    # Create grid with extended bounds
    lats = np.linspace(extended_lat_bounds[0], extended_lat_bounds[1], grid_resolution)
    lons = np.linspace(extended_lon_bounds[0], extended_lon_bounds[1], grid_resolution)

    tasks = [(lat, lon, start_date, end_date) for lat in lats for lon in lons]
    total_points = len(tasks)
    logging.info(f"Number of points to process: {total_points}")
    logging.info(f"Date range: {start_date} to {end_date}")

    binned_data = {}
    # Process in parallel with a progress bar
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_args, t) for t in tasks]

        # Display a progress bar for the completed tasks
        for f in tqdm(as_completed(futures), total=total_points, desc="Processing grid points"):
            lat, lon, nightly_averages = f.result()
            loc_key = f"{lat}_{lon}"
            binned_data[loc_key] = nightly_averages

    logging.info("All points processed. Converting to DataFrame and computing seasonal averages.")

    # Convert binned_data to a DataFrame
    rows = []
    for loc_key, nights in binned_data.items():
        lat_str, lon_str = loc_key.split("_")
        lat_val = float(lat_str)
        lon_val = float(lon_str)
        for night_str, vals in nights.items():
            avg_val = vals["average"]
            date_val = datetime.fromisoformat(night_str).date()
            rows.append((lat_val, lon_val, date_val, avg_val))

    df_all = pd.DataFrame(rows, columns=["lat", "lon", "date", "average"])
    # Determine season
    df_all["season"] = df_all["date"].apply(determine_season)

    # Compute overall averages (all data)
    df_overall = df_all.groupby(["lat", "lon"])["average"].mean().reset_index()

    # Compute seasonal averages across years
    df_seasonal = df_all.groupby(["lat", "lon", "season"])["average"].mean().reset_index()

    # Save the raw nightly averages to JSON if logging is enabled
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

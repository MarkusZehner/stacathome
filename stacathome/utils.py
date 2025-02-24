import multiprocessing
import time
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from odc.geo.geobox import BoundingBox
from pyproj import Proj, Transformer
from pystac import Item
from shapely import Point, Polygon


def time_range_parser(time_range):
    if isinstance(time_range, int):
        return str(time_range)
    elif isinstance(time_range, (tuple, list)):
        if len(time_range) != 2:
            raise ValueError("Time range must be a tuple or list of length 2")
        return f"{time_range[0]}/{time_range[1]}"
    else:
        raise ValueError("Time range must be an integer or a tuple or list of integers")


def cut_box_to_edge_around_coords(point, gbox, n_pix_edge):
    y_center = np.argmin(np.abs(gbox.coordinates["y"].values - point.y))
    x_center = np.argmin(np.abs(gbox.coordinates["x"].values - point.x))

    min_y = y_center - (n_pix_edge // 2)
    min_x = x_center - (n_pix_edge // 2)

    return gbox[
        min_y : min_y + n_pix_edge,
        min_x : min_x + n_pix_edge,
    ]


def get_utm_crs_from_lon_lat(lon, lat):
    utm_zone = int(np.floor(lon + 180) / 6) + 1
    # Handle Norway 32V anomaly
    if (56 <= lat < 64) and (3 <= lon < 6):
        utm_zone = 32
    # Handle Svalbard zones
    if 72 <= lat < 84:
        if 0 <= lon < 9:
            utm_zone = 31
        elif 9 <= lon < 21:
            utm_zone = 33
        elif 21 <= lon < 33:
            utm_zone = 35
        elif 33 <= lon < 42:
            utm_zone = 37

    hemisphere = 700 if lat < 0 else 600
    return 32000 + hemisphere + utm_zone


def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = int(degrees) + int(minutes) / 60 + float(seconds) / 3600
    if direction in ['S', 'W']:
        decimal *= -1
    return decimal


def parse_dms_to_lon_lat_point(dms_string):
    dms_string = dms_string.replace('°', '-').replace('\'', '-').replace('"', '-')
    dms_string = dms_string.split(' ')
    lat = dms_to_decimal(*dms_string[0].split('-'))
    lon = dms_to_decimal(*dms_string[1].split('-'))
    return Point(lon, lat)


def parse_dec_to_lon_lat_point(dec_string):
    lat, lon = map(float, dec_string.split(','))
    return Point(lon, lat)


def get_time_binning(
    timeframe: int | tuple[int] | tuple[str],
    start_of_time: str = '1980-01-01',
    end_of_time: str = '2030-01-01',
    t_freq: str = '15D',
):
    """
    Generate a common time binning and return periods of interest.
    """
    # add other timeframe selection options?
    time_bins = pd.date_range(start=start_of_time, end=end_of_time, freq=t_freq)
    if isinstance(timeframe, int) and timeframe in time_bins.year:
        return time_bins.where(time_bins.year == timeframe).dropna()
    elif isinstance(timeframe, tuple) and len(timeframe) == 2 and all(year in time_bins.year for year in timeframe):
        return time_bins.where((time_bins.year >= timeframe[0]) & (time_bins.year <= timeframe[1])).dropna()
    elif isinstance(timeframe, tuple) and isinstance(timeframe[0], str):
        return time_bins.where((time_bins >= timeframe[0]) & (time_bins <= timeframe[1])).dropna()
    else:
        return time_bins


def filter_items_to_data_coverage(items: list[Item], bbox: BoundingBox, sensor: str = "S2") -> list[Item]:
    """
    Using the data coverage geometry in the STAC item to filter the items and remove
    those that do not intersect with the requested bounds.
    also sorts the items by ascending datetime.

    Parameters
    ----------
    items : list[Item]
        List of pystac.Item objects.
    bbox : BoundingBox
        The bounding box to filter the items by.
    sensor : str, optional
        The sensor to filter the items by, by default "S2".
    """
    sort_by = {
        'sentinel-2-l2a': "datetime",
        'modis-13Q1-061': "start_datetime",
        'esa-worldcover': "start_datetime",
    }
    if sensor not in sort_by.keys():
        raise ValueError("Sensor not (yet) supported.")

    items_within = []
    for i in range(len(items)):
        if Polygon(*items[i].geometry["coordinates"]).contains_properly(bbox):
            items_within.append(items[i])
    if len(items_within) > 0:
        items_within = sorted(items_within, key=lambda x: x.properties[sort_by[sensor]])
    return items_within


def run_with_multiprocessing(target_function: Callable, **func_kwargs):
    """
    Wrapper to execute a function in separate processes to isolate memory leaks.
    will retry the function 3 times if it fails.

    Args:
        target_function (callable): The function to run in a separate process.
        num_runs (int): The number of times to execute the function in separate processes.
        func_args (tuple): Positional arguments for the target function.
        func_kwargs (dict): Keyword arguments for the target function.
    """
    for _ in range(3):
        process = multiprocessing.Process(target=target_function, kwargs=func_kwargs, name="throwawayWorker")

        process.start()
        process.join()
        if process.exitcode is None:
            print("Process Worker did not exit cleanly, terminating...")
            process.terminate()

        print(f"Process Worker completed with exit code {process.exitcode}")
        if process.exitcode == 0:
            break

    time.sleep(0.1)


def get_transform(from_crs, to_crs, always_xy=True):
    if isinstance(from_crs, int) or isinstance(from_crs, str) and len(from_crs) < 6:
        from_crs = Proj(f"epsg:{from_crs}")
    if isinstance(to_crs, int) or isinstance(to_crs, str) and len(to_crs) < 6:
        to_crs = Proj(f"epsg:{to_crs}")

    project = Transformer.from_proj(
        from_crs,  # source coordinate system
        to_crs,
        always_xy=always_xy,
    )
    return partial(transform_coords, project=project)


def transform_coords(x_y, project):
    for i in range(len(x_y)):
        x_y[i] = project.transform(x_y[i][0], x_y[i][1])
    return x_y

import math
import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from urllib.request import urlretrieve

import re
import zipfile
from collections import Counter
import zarr
import planetary_computer as pc
from pyproj import Proj, Transformer
from shapely import box, unary_union, transform, Point


def create_utm_grid_bbox(bbox, grid_size=60):
    """
    Snap the bounding box to a utm grid of the specified pixel size.

    Parameters:
    ----------
    bbox (BoundingBox): The bounding box to snap.
        grid_size (int): The size of the grid to snap to.

    Returns:
    -------
    shapely.geometry.box: The snapped bounding box.
    """
    xmin, ymin, xmax, ymax = bbox
    xmin_snapped = math.floor(xmin / grid_size) * grid_size
    ymin_snapped = math.floor(ymin / grid_size) * grid_size
    xmax_snapped = math.ceil(xmax / grid_size) * grid_size
    ymax_snapped = math.ceil(ymax / grid_size) * grid_size
    return box(xmin_snapped, ymin_snapped, xmax_snapped, ymax_snapped)


def arange_bounds(bounds, step):
    return np.arange(bounds[0], bounds[2], step), np.arange(bounds[1], bounds[3], step)


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


def compute_scale_and_offset(da, n=16):
    """Calculate offset and scale factor for int conversion

    Based on Krios101's code above.
    """

    vmin = np.nanmin(da).item()
    vmax = np.nanmax(da).item()

    # stretch/compress data to the available packed range
    # -2 to reserve the upper bit for nan only, otherwise maxval == nan
    scale_factor = (vmax - vmin) / (2 ** n - 2)

    # # translate the range to be symmetric about zero
    # add_offset = (vmin + 2 ** (n - 1) * scale_factor) + mean_shift

    return scale_factor  # , add_offset


def get_asset(href: str, save_path: Path, signer: callable = pc.sign):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        return
    try:
        urlretrieve(signer(href), save_path)
    except (KeyboardInterrupt, SystemExit):
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception as e:
                print(f"Error during cleanup of file {save_path}:", e)
    except Exception as e:
        print(f"Error downloading {href}:", e)


def download_assets_parallel(asset_list, max_workers=4, signer: callable = pc.sign):
    get_asset_with_sign = partial(get_asset, signer=signer)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda args: get_asset_with_sign(*args), asset_list)


def cube_to_zarr_zip(path, data):
    store = zarr.ZipStore(path, mode="x", compression=zipfile.ZIP_BZIP2)
    data.to_zarr(store, mode="w-", consolidated=True)
    store.close()


def most_common(lst):
    return Counter(lst).most_common(1)[0][0]


def resolve_best_containing(items):
    containing = [i for i in items if i[2]]
    if not containing:
        return None
    if len(containing) == 1:
        return containing[0]
    min_dist = np.argmin(i[3] for i in containing)
    return containing[min_dist]


def merge_to_cover(items, target_shape):
    best_crs = max(items, key=lambda x: x[4])[1]
    candidates = sorted(items, key=lambda i: i[4], reverse=True)
    merged = []
    merged_shapes = []
    for item in candidates:
        tr_shape = transform(item[5], get_transform(item[1], best_crs))
        merged.append(item)
        merged_shapes.append(tr_shape)
        if unary_union(merged_shapes).contains(target_shape):
            break

    return merged


def is_valid_partial_date_range(s):
    pattern = r"^\d{4}(?:-\d{2}){0,2}(?:/\d{4}(?:-\d{2}){0,2})?$"
    return bool(re.match(pattern, s))


def parse_time(t_index: str | int | list | tuple | dict):
    """
    helper for several usecases:

    - str: request range as YYYY-MM-DD/YYYY-MM-DD with optional month, day,
            or selected date by YYYY-MM-DD with optional month, day

    - single int of a year: request the whole year of data
    - list of ints: request the listed years
    - tuple of two ints: request the inclusive range of years
    - tuple (year, month): request specific month(1-12) of a year(>1900)
    - list of datetime objects: handled as time bins to separate the download between

    Raises:
        NotImplementedError: _description_
    """
    if isinstance(t_index, str):
        if is_valid_partial_date_range(t_index):
            return t_index
        else:
            raise ValueError("String should be in format of YYYY-MM-DD/YYYY-MM-DD or valid subset.")

    elif isinstance(t_index, int):
        return f"{str(t_index)}-01-01/{str(t_index)}-12-31"

    elif isinstance(t_index, list[int]):
        return [f"{str(t)}-01-01/{str(t)}-12-31" for t in sorted(set(t_index))]

    elif isinstance(t_index, tuple) and len(t_index) == 2:
        if isinstance(t_index[0], int) and isinstance(t_index[1], int):
            if min(t_index) > 1900:
                return f"{str(min(t_index))}-01-01/{str(max(t_index))}-12-31"
            elif min(t_index) <= 12:
                return f"{str(max(t_index))}-{str(min(t_index))}"
        else:
            raise ValueError("Tuple should be of length 2 with integers")

    raise NotImplementedError


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

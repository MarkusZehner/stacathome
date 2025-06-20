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
from shapely import box, unary_union, transform, Point, buffer
from pystac import Item

from datetime import timedelta


def merge_item_datetime_by_timedelta(items: list[Item], max_t_diff: timedelta = None):
    items_mod = items.copy()
    if max_t_diff is None:
        max_t_diff = timedelta(minutes=10)
    t_diffs = np.diff([i.datetime for i in items])
    for i, diff in enumerate(t_diffs):
        if diff < max_t_diff:
            items_mod[i + 1].datetime = items_mod[i].datetime
    return items_mod


def create_utm_grid_bbox(bbox, grid_size=60, offset_x=0, offset_y=0):
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
    xmin_snapped = math.floor(xmin / grid_size) * grid_size + offset_x
    ymin_snapped = math.floor(ymin / grid_size) * grid_size + offset_y
    xmax_snapped = math.ceil(xmax / grid_size) * grid_size + offset_x
    ymax_snapped = math.ceil(ymax / grid_size) * grid_size + offset_y
    return box(xmin_snapped, ymin_snapped, xmax_snapped, ymax_snapped)


def arange_bounds(bounds, step):
    """
    Creates two arrays from the bounds with given step size.

    Parameters:
    ----------
    bounds : list
        containing four float values [xmin, ymin, xmax, ymax]
    step : float
        the step size for the arrays

    Returns:
    -------
    (list, list): 
        Two arrays representing the x and y coordinates
    """
    return np.arange(bounds[0], bounds[2], step), np.arange(bounds[1], bounds[3], step)


def get_transform(from_crs, to_crs, always_xy=True):
    """
    Get a transformer function to convert coordinates from one CRS to another.
    Parameters:
    ----------
    from_crs : int, str, or Proj
        The source coordinate reference system (CRS) identifier or Proj object.
    to_crs : int, str, or Proj
        The target coordinate reference system (CRS) identifier or Proj object.
    always_xy : bool, default True
        If True, always treat the first coordinate as x and the second as y.
        If False, the order depends on the CRS.
    Returns:
    -------
    function
        A function that takes a list of (x, y) tuples and transforms them from the source CRS to the target CRS.
    """
    if isinstance(from_crs, int) or isinstance(from_crs, str) and len(from_crs) < 6:
        from_crs = Proj(f"epsg:{from_crs}")
    if isinstance(to_crs, int) or isinstance(to_crs, str) and len(to_crs) < 6:
        to_crs = Proj(f"epsg:{to_crs}")

    project = Transformer.from_proj(
        from_crs,  # source coordinate system
        to_crs,
        always_xy=always_xy,
    )
    return partial(__transform_coords, project=project)


def __transform_coords(x_y: list[tuple[float, float]], project: Transformer):
    """
    Helper to get a transformer function to convert coordinates from one CRS to another.

    Parameters:
    ----------
    x_y : list(tuple(float, float)
        A list of tuples containing (x, y) coordinates to be transformed.
    project : pyproj.Transformer
        A pyproj Transformer object that defines the transformation from one CRS to another.
    Returns:
    -------
    list(tuple(float, float))
        A list of transformed (x, y) coordinates.
    """
    for i in range(len(x_y)):
        x_y[i] = project.transform(x_y[i][0], x_y[i][1])
    return x_y


def compute_scale_and_offset(da, n=16):
    """
    Calculate offset and scale factor for int conversion
    (taken from vitus or claires codebase)
    Based on Krios101's code above.

    Parameters:
    ----------
    da : xarray.DataArray
        The data array for which to compute the scale and offset.
    n : int, default 16
        The number of bits to use for the packed representation.

    Returns:
    -------
    float
        The scale factor for converting the data array to a packed representation.
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
    """
    Get one asset from a given href and save it to the specified path.
    This function will create the necessary directories if they do not exist,
    and will skip downloading if the file already exists.
    It also handles cleanup in case of an interruption during the download.

    Parameters:
    ----------
    href : str
        The URL of the asset to download.
    save_path : Path
        The local path where the asset should be saved.
    signer : callable, default pc.sign
        A function to sign the URL if needed (e.g., for accessing protected resources).

    Returns:
    -------
    None

    Raises:
    -------
    Exception: If there is an error during the download process.
    KeyboardInterrupt: If the download is interrupted by the user.
    SystemExit: If the download is interrupted by a system exit.
    """
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
    """
    Download a list of assets in parallel using a thread pool executor.
    This function will create a partial function with the signer and then use
    a thread pool to download each asset concurrently.

    Parameters:
    ----------
    asset_list : list of tuples
        A list where each tuple contains the href and the save path for the asset.
        Example: [(href1, save_path1), (href2, save_path2), ...]
    max_workers : int, default 4
        The maximum number of worker threads to use for downloading.
    signer : callable, default pc.sign
        A function to sign the URL if needed (e.g., for accessing protected resources).

    Returns:
    -------
    None
    """
    get_asset_with_sign = partial(get_asset, signer=signer)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda args: get_asset_with_sign(*args), asset_list)


def cube_to_zarr_zip(path, data):
    """Stores a xarray DataArray or Dataset to a zarr zip file.

    Parameters:
    ----------
    path : str or Path
        The path where the zarr zip file will be saved.
    data : xarray.Dataset
    """
    store = zarr.storage.ZipStore(path, mode="w", compression=zipfile.ZIP_BZIP2)
    data.to_zarr(store, mode="w", consolidated=True)
    store.close()


def most_common(lst):
    """Returns the (first) most common element in a list.
    Parameters:
    ----------
    lst : list
        The list from which to find the most common element.
    Returns:
    -------
    The most common element in the list.
    """
    return Counter(lst).most_common(1)[0][0]


def resolve_best_containing(items):
    """helper to check coverage of satellite items with a given area.
    items are tuples of (item_id, x, contains_flag, x, area, x).

    Parameters:
    ----------
    items : list of tuples
        Each tuple contains (item_id, x, contains_flag, x, area, x).
        The `contains_flag` is a boolean indicating whether the item contains other items.
    Returns:
    -------
    tuple or None
        The item that contains the search area with the smallest distance between both centroids.
    """

    containing = [i for i in items if i[2]]
    if not containing:
        return None
    if len(containing) == 1:
        return containing[0]
    min_dist = np.argmin(i[3] for i in containing)
    return containing[min_dist]


def merge_to_cover(items, target_shape):
    """helper to check coverage of satellite items with a given area.

    will merge items until the target shape is covered.
    iterating over the items by descending instersection area with request.

    Parameters:
    ----------
    items : list of tuples
        Each tuple contains (item_id, crs, x, x, area, shapely.box).
        The `contains_flag` is a boolean indicating whether the item contains other items.
    target_shape : shapely.geometry.Polygon or shapely.geometry.box
        The target shape to check for coverage.
    Returns:
    -------
    list
        A list of items that cover the target shape, sorted by their area in descending order.
    """
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


def metric_buffer(shape, distance: int, return_box=False, crs=4326):
    """creates a metric distance buffer around a shape.
    This function transforms the shape to a UTM coordinate system based on its centroid,
    applies a buffer of the specified distance, and then transforms it back to the original CRS.
    If `return_box` is True, it returns a bounding box of the buffered shape.
    If `return_box` is False, it returns the buffered shape itself.
    Parameters:
    ----------
    shape (any) shapely geometry : 
        The shape to buffer.
    distance : int
        The distance in meters to buffer the shape.
    return_box : bool, optional
        If True, returns a bounding box of the buffered shape. Defaults to False.
    crs : int, optional
        The coordinate reference system code to use for the transformation. Defaults to 4326.
    """
    shape_center = shape.centroid
    crs_code = get_utm_crs_from_lon_lat(shape_center.x, shape_center.y)
    shape = transform(shape, get_transform(crs, crs_code))
    shape = transform(buffer(shape, distance=distance), get_transform(crs_code, crs))
    if return_box:
        return box(*shape.bounds)
    else:
        return shape


def is_valid_partial_date_range(s):
    """Regex to check if a string is a valid partial date range.
    The string should be in the format YYYY-MM-DD/YYYY-MM-DD with optional month, day or second date.

    Parameters:
    ----------
    s : str
        The string to check.

    Returns:
    -------
    bool
        True if the string is a valid partial date range, False otherwise.
    """
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
    """
    Get the S2 UTM grid crs code based on longitude and latitude.
    The S2 UTM grid has some pecularities in Norway and Svalbard,
    which this function accounts for.

    Parameters:
    ----------
    lon : float
        The longitude in degrees.
    lat : float
        The latitude in degrees.
    Returns:
    -------
    int
        The EPSG code for the UTM CRS.
    """
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

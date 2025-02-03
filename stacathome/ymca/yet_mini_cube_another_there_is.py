import copy
import json
import os
import shutil
import sys
import multiprocessing
import time
import datetime
from itertools import islice

import numpy as np
import xarray as xr
import zarr
import pandas as pd
import geopandas as gpd
import planetary_computer as pc
from odc import stac
from odc.geo.geobox import BoundingBox, GeoBox
from stac_geoparquet import to_item_collection
from pystac import Item
from pystac_client import Client
from pystac_client.exceptions import APIError
from shapely import transform
from shapely.geometry import Point, box, Polygon
from rasterio.errors import WarpOperationError, RasterioIOError
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
from pyproj import CRS

from stacathome.utils import get_transform

# STAC query
def request_data_by_tile(
    tile_id: str,
    start_time: str,
    end_time: str,
    url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    collection: str = "sentinel-2-l2a",
    save_dir: str = None,
) -> list[Item]:
    """
    Requesting items from STAC-API by the tile_id, start_time and end_time,
    utilizing a filter in the query.

    If save_dir is provided, the query results are written to json.
    If a file with matching naming scheme
    (f"{collection}_{tile_id}_{start_time}_{end_time}_query.json")
    exists in the save_dir,
    the query is skipped and its contents are loaded instead.

    Parameters
    ----------
    tile_id : str
        The tile_id to filter the query by.
    start_time : str
        The start time of the query.
    end_time : str
        The end time of the query.
    url : str, optional
        The url of the STAC-API, by default "https://planetarycomputer.microsoft.com/api/stac/v1".
    collection : str, optional
        The collection to query, by default "sentinel-2-l2a".
    save_dir : str, optional
        The directory to save the query results to, by default None.

    Returns
    -------
    list
        A list of pystac.Item objects.
    """

    out_path = (
        os.path.join(
            save_dir, f"{collection}_{tile_id}_{start_time}_{end_time}_query.json"
        )
        if save_dir
        else None
    )
    if out_path and os.path.exists(out_path):
        with open(out_path, "r") as json_file:
            query_dict = json.load(json_file)
    else:
        if collection == "sentinel-2-l2a":
            filter_arg = "s2:mgrs_tile"
        elif collection == "modis-13Q1-061":
            filter_arg = "modis:tile-id"
        else:
            raise ValueError("Collection not (yet) supported.")

        catalog = Client.open(url)
        query = catalog.search(
            collections=[collection],
            datetime=f"{start_time}/{end_time}",
            query={filter_arg: dict(eq=tile_id)},
        )

        for _ in range(3):  # retry 3 times
            try:
                query_dict = query.item_collection_as_dict()
                break
            except APIError as e:
                query_dict = {}
                print(f"Error: {e}", flush=True)
            time.sleep(1)

        if query_dict == {}:
            print(
                f"Failed to get data for {tile_id} {start_time} {end_time}.", flush=True
            )
            return []

        if out_path:
            with open(out_path, "w") as json_file:
                json.dump(query_dict, json_file, indent=4)

    return [Item.from_dict(feature) for feature in query_dict["features"]]


# download and save data
# TODO: merge S2 and MODIS functions into one?
def S2_cube_part_in_native_res(
    items: list[Item],
    out_path: str,
    request_geobox: GeoBox,
    sel_bands: list[str],
    asset_bin_size: int = 50,
) -> None:
    """
    Processing the items into cubes in size of the request_geobox, stored in out_path.
    Uses asset_bin_size to subset large amounts of items into smaller batches to save.

    Parameters
    ----------
    items : list[Item]
        List of pystac.Item objects.
    out_path : str
        The path to save the zipped zarr cubes to.
    request_geobox : GeoBox
        The geobox to request the data in.
    sel_bands : list[str]
        The bands to include in the cube.
    asset_bin_size : int, optional
        The size of the asset batch, by default 50.

    Returns
    -------
    None
    """
    if len(items) == 0:
        return

    _bands_10m = {"B02", "B03", "B04", "B08", "SCL"}
    _bands_20m = {"B05", "B06", "B07", "B8A", "B11", "B12"}
    _bands_60m = {"B01", "B09", "B10"}
    _s2_dytpes = "uint16"
    bounds = list(request_geobox.boundingbox)
    crs = request_geobox.crs
    transformed_box = transform(
        box(*bounds), get_transform(str(crs).split(":")[1], 4326)
    )
    items_with_data_coverage = __filter_items_to_data_coverage(
        items, transformed_box, sensor="S2"
    )

    print(  # TODO: to be replaced with logging
        f"Processing {len(items_with_data_coverage)} items for {out_path}", flush=True
    )

    box_10m = GeoBox.from_bbox(bounds, crs, resolution=10)
    box_20m = GeoBox.from_bbox(bounds, crs, resolution=20)
    box_60m = GeoBox.from_bbox(bounds, crs, resolution=60)

    sel_bands_s = set(sel_bands)
    bands_10m = list(_bands_10m & sel_bands_s)
    bands_20m = list(_bands_20m & sel_bands_s)
    bands_60m = list(_bands_60m & sel_bands_s)

    for bands, gbox in zip(
        [bands_10m, bands_20m, bands_60m], [box_10m, box_20m, box_60m]
    ):
        if len(bands) == 0:
            continue

        for time_bin in range(0, len(items_with_data_coverage), asset_bin_size):
            _out_path = (  # adding resolution and item bin to the basename
                out_path
                + f"_{int(gbox.resolution.x)}m_{str(time_bin).zfill(4)}_{str(time_bin+asset_bin_size).zfill(4)}.zarr.zip"
            )
            if os.path.exists(_out_path):
                print(  # TODO: to be replaced with logging
                    f"Skipping {time_bin}-{time_bin+asset_bin_size} for {out_path}",
                    flush=True,
                )
                continue

            for _ in range(3):  # retry 3 times
                try:
                    start_time = time.time()
                    save_stac_to_zarr_zip(
                        items_with_data_coverage[time_bin : time_bin + asset_bin_size],
                        _out_path,
                        bands,
                        _s2_dytpes,
                        geobox=gbox,
                    )
                    print(  # TODO: to be replaced with logging
                        f"Time taken for {time_bin}-{time_bin+asset_bin_size} for {out_path}: {time.time()-start_time:.0f} s",
                        flush=True,
                    )
                    break
                except (WarpOperationError, RasterioIOError) as e:
                    print(f"Error creating Zarr: {e} deleting {_out_path}", flush=True)
                    try:
                        os.remove(_out_path)
                        print(
                            f"Deleted due to Error: {_out_path}", flush=True
                        )  # TODO: to be replaced with logging
                    except OSError as e:
                        print(
                            "Error: %s : %s" % (_out_path, e.strerror), flush=True
                        )  # TODO: to be replaced with logging

                time.sleep(1)

# download and save data
def MODIS_cube_part(  # TODO: enable geobox (resampled) or boundingbox(native) as target area
    items: list[Item],
    out_path: str,
    request_geobox: GeoBox,
    sel_bands: list[str],
    asset_bin_size: int = 50,
) -> None:
    """
    Processing the items into cubes in size of the request_geobox, stored in out_path.
    Uses asset_bin_size to subset large amounts of items into smaller batches to save.

    Parameters
    ----------
    items : list[Item]
        List of pystac.Item objects.
    out_path : str
        The path to save the zipped zarr cubes to.
    request_geobox : GeoBox
        The geobox to request the data in.
    sel_bands : list[str]
        The bands to include in the cube.
    asset_bin_size : int, optional
        The size of the asset batch, by default 50.

    Returns
    -------
    None
    """
    if len(items) == 0:
        return

    _bands_MODIS = {
        "250m_16_days_EVI",
        "250m_16_days_NDVI",
        "250m_16_days_VI_Quality",
        "250m_16_days_MIR_reflectance",
        "250m_16_days_NIR_reflectance",
        "250m_16_days_red_reflectance",
        "250m_16_days_blue_reflectance",
        "250m_16_days_sun_zenith_angle",
        "250m_16_days_pixel_reliability",
        "250m_16_days_view_zenith_angle",
        "250m_16_days_relative_azimuth_angle",
    }
    _MODIS_dytpes = {
        "250m_16_days_EVI": "int16",
        "250m_16_days_NDVI": "int16",
        "250m_16_days_VI_Quality": "uint16",
        "250m_16_days_MIR_reflectance": "int16",
        "250m_16_days_NIR_reflectance": "int16",
        "250m_16_days_red_reflectance": "int16",
        "250m_16_days_blue_reflectance": "int16",
        "250m_16_days_sun_zenith_angle": "int16",
        "250m_16_days_pixel_reliability": "int16",
        "250m_16_days_view_zenith_angle": "int16",
        "250m_16_days_relative_azimuth_angle": "int16",
    }
    bounds = list(request_geobox.boundingbox)
    crs = request_geobox.crs
    bbox_bounds = box(*bounds)
    bbox_bounds.buffer(250)
    transformed_box = transform(
        bbox_bounds, get_transform(str(request_geobox.crs).split(":")[1], 4326)
    )
    items_with_data_coverage = __filter_items_to_data_coverage(
        items, transformed_box, sensor="MODIS"
    )
    print(  # TODO: to be replaced with logging
        f"Processing {len(items_with_data_coverage)} items for {out_path}", flush=True
    )

    bands_MODIS = list(_bands_MODIS & set(sel_bands))
    box_20m = GeoBox.from_bbox(bounds, crs, resolution=20)

    if len(bands_MODIS) == 0:
        return

    for time_bin in range(0, len(items), asset_bin_size):
        _out_path = (
            out_path
            + f"_{str(time_bin).zfill(4)}_{str(time_bin+asset_bin_size).zfill(4)}.zarr.zip"
        )
        if os.path.exists(_out_path):
            print(  # TODO: to be replaced with logging
                f"Skipping {time_bin}-{time_bin+asset_bin_size} for {out_path}",
                flush=True,
            )
            continue

        for _ in range(3):  # retry 3 times
            try:
                start_time = time.time()
                save_stac_to_zarr_zip(
                    items_with_data_coverage[time_bin : time_bin + asset_bin_size],
                    _out_path,
                    bands_MODIS,
                    _MODIS_dytpes,
                    geobox=box_20m,
                )
                print(  # TODO: to be replaced with logging
                    f"Time taken for {time_bin}-{time_bin+asset_bin_size} for {out_path}: {time.time()-start_time:.0f} s",
                    flush=True,
                )
            except (WarpOperationError, RasterioIOError) as e:
                print(
                    f"Error creating Zarr: {e} deleting {_out_path}", flush=True
                )  # TODO: to be replaced with logging
                try:
                    shutil.rmtree(_out_path)
                except OSError as e:
                    print(
                        "Error: %s : %s" % (_out_path, e.strerror), flush=True
                    )  # TODO: to be replaced with logging
            time.sleep(1)

# download and save data
def save_stac_to_zarr_zip(
    items: list[Item],
    out_path: str,
    bands: list[str],
    dtype: str | dict,
    geobox: GeoBox = None,
    boundingbox: BoundingBox = None,
    resampling: str | dict = "nearest",
) -> None:
    """
    Downloading the items and saving them to a zarr file.
    Uses planetary_computer.sign to sign the request.
    *Potential memory leak* when loading the data.

    Parameters
    ----------
    items : list[Item]
        List of pystac.Item objects.
    out_path : str
        The path to save the zipped zarr cubes to.
    bands : list[str]
        The bands to include in the cube.
    dtype : str | dict
        The datatype of the bands.
    geobox : GeoBox, optional
        The geobox to request the data in, by default None.
    boundingbox : BoundingBox, optional
        The bounding box to request the data in, by default None.
    resampling : str | dict, optional
        The resampling method to use, by default "nearest".

    Returns
    -------
    None
    """
    parameters = {
        "items": items,
        "patch_url": pc.sign,
        "bands": bands,
        "dtype": dtype,
        #'chunks': {"time": -1, "x": -1, "y": -1},  # setting chunks to None diables Dask, but was slower in testing
        "groupby": "solar_day",
        "resampling": resampling,
        "fail_on_error": True,
    }
    if geobox:
        parameters["geobox"] = geobox
    if boundingbox:
        parameters["bbox"] = boundingbox

    if not geobox and not boundingbox:
        raise ValueError("Either geobox or boundingbox must be provided.")

    store = zarr.ZipStore(out_path, mode="w")
    data = stac.load(**parameters).compute()
    data.to_zarr(store, mode="w")
    store.close()
    del data


# utility
def __filter_items_to_data_coverage(
    items: list[Item], bbox: BoundingBox, sensor: str = "S2"
) -> list[Item]:
    """
    Using the data coverage geometry in the STAC item to filter the items and remove
    those that do not intersect with the requested bounds.

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
        "S2": "datetime",
        "MODIS": "start_datetime",
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


# TODO: drop/adapt these functions when v0 is fixed
def __filter_for_existing_cubes(existing_cubes, locations, sensor="S2"):
    """
    function to handle inconsistent downloaded data, to be removed when v0 is fixed.
    """
    existing_dict = {}
    for i in existing_cubes:
        if not i.endswith(".zarr.zip"):
            continue

        i = i.split("_")
        # first version cubes
        if i[3] == sensor:
            loc_id = "_".join(i[1:3])
            year = i[5].split("-")[0]
            if loc_id not in existing_dict:
                existing_dict[loc_id] = [int(year)]
            else:
                existing_dict[loc_id].append(int(year))

        # # second version cubes
        # elif i[4]==sensor:
        #     loc_id = "_".join(i[2:4])
        #     year_1 = int(i[6].split("-")[0])
        #     year_2 = int(i[7].split("-")[0])
        #     years = list(range(year_1, year_2+1))
        #     if loc_id not in existing_dict:
        #         existing_dict[loc_id] = years
        #     else:
        #         existing_dict[loc_id].append(years)
    years = list(range(2014, 2026))
    finished_new = {}
    unfinished_new = {}
    for location in locations:
        present_cubes = [
            filename for filename in existing_cubes if location in filename
        ]
        present_cubes = [
            filename for filename in present_cubes if f"_{sensor}_" in filename
        ]
        for y in years:
            present_cubes_y = [
                filename
                for filename in present_cubes
                if str(y) in filename.split("_")[6]
            ]
            if len(present_cubes_y) == 0:
                continue

            parts = None
            if sensor == "S2":
                present_cubes_y_10m = [
                    filename for filename in present_cubes_y if "_10m" in filename
                ]
                present_cubes_y_20m = [
                    filename for filename in present_cubes_y if "_20m" in filename
                ]
                present_cubes_y_60m = [
                    filename for filename in present_cubes_y if "_60m" in filename
                ]

                parts = present_cubes_y_10m[0].split("_")

                if (
                    len(present_cubes_y_10m)
                    == len(present_cubes_y_20m)
                    == len(present_cubes_y_60m)
                ):
                    if len(present_cubes_y_10m) == 0:
                        print(f"Error: {location} {y} {present_cubes_y}", flush=True)

                    if location not in finished_new:
                        finished_new[location] = [(parts[6], parts[7])]
                    else:
                        finished_new[location].extend([(parts[6], parts[7])])
                else:
                    if location not in unfinished_new:
                        unfinished_new[location] = [
                            (
                                present_cubes_y_10m,
                                present_cubes_y_20m,
                                present_cubes_y_60m,
                            )
                        ]
                    else:
                        unfinished_new[location].extend(
                            [
                                (
                                    present_cubes_y_10m,
                                    present_cubes_y_20m,
                                    present_cubes_y_60m,
                                )
                            ]
                        )

            if not parts:
                parts = present_cubes_y[0].split("_")

            if location not in existing_dict:
                existing_dict[location] = [
                    i
                    for i in range(
                        int(parts[6].split("-")[0]), int(parts[7].split("-")[0]) + 1
                    )
                ]
            else:
                [
                    existing_dict[location].append(i)
                    for i in range(
                        int(parts[6].split("-")[0]), int(parts[7].split("-")[0]) + 1
                    )
                ]

    return existing_dict, unfinished_new


# TODO: drop/adapt these functions when v0 is fixed
def __filter_years_to_process(
    locations, existing_dict, year_range_process=(2014, 2026)
):  
    """
    function to handle inconsistent downloaded data, to be removed when v0 is fixed.
    """
    years_to_process = set(range(*year_range_process))
    for location in locations.keys():
        location_id_old = "_".join(location.split("_")[1:3])
        location_id_new = "_".join(location.split("_")[0:4])

        if location_id_old in existing_dict and not location_id_new in existing_dict:
            years_remain_to_process = list(
                years_to_process - set(existing_dict[location_id_old])
            )
            years_remain_to_process.sort()
            locations[location].append(
                __find_consecutive_sequences(list(years_remain_to_process))
            )

        if location_id_new in existing_dict and not location_id_old in existing_dict:
            years_remain_to_process = list(
                years_to_process - set(existing_dict[location_id_new])
            )
            years_remain_to_process.sort()
            locations[location].append(
                __find_consecutive_sequences(list(years_remain_to_process))
            )

        if location_id_old in existing_dict and location_id_new in existing_dict:
            years_remain_to_process = list(
                years_to_process
                - set(existing_dict[location_id_old])
                - set(existing_dict[location_id_new])
            )
            years_remain_to_process.sort()
            locations[location].append(
                __find_consecutive_sequences(list(years_remain_to_process))
            )

        if (
            location_id_old not in existing_dict
            and location_id_new not in existing_dict
        ):
            locations[location].append(
                __find_consecutive_sequences(list(years_to_process))
            )

    return locations


# TODO: drop/adapt these functions when v0 is fixed
def __find_consecutive_sequences(numbers:list) -> list:
    """
    Find sequences of consecutive numbers in a list.
    """
    if not numbers:
        return []

    numbers = sorted(numbers)  # Ensure the list is sorted
    sequences = []
    current_sequence = [numbers[0]]

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:  # Check if consecutive
            current_sequence.append(numbers[i])
        else:
            sequences.append(current_sequence)  # Save the completed sequence
            current_sequence = [numbers[i]]  # Start a new sequence

    sequences.append(current_sequence)  # Add the last sequence
    return sequences


# utility
def get_rs_product_tiles_for_locs(
    locations_csv_path: str,
    grid_parquet_path: str,
    target_res_spatial: int = 20,
    spatial_size: int = 128,
    coarsest_res_sensor: int = 60,
    filter_by: str = None,
) -> dict:
    """
    Given a CSV with 'SITE_ID', 'latitude' and 'longitude' columns, and a grid parquet file with 'tile', 'utm_bounds' and 'epsg' columns,
    this function returns a dictionary with the location name as key and the corresponding tile, epsg and geobox as values.

    an additional filter_by geopackage can be provided to filter the locations.

    BUG: the function will round the selected spatial size to the nearest multiple of the coarsest resolution.
    128 (pix) at target size 20 (m) will be rounded to 126 (pix).


    Parameters
    ----------
    locations_csv_path : str
        The path to the CSV file with the locations.
    grid_parquet_path : str
        The path to the grid parquet file.
    target_res_spatial : int, optional
        The target spatial resolution, by default 20 (m).
    spatial_size : int, optional
        The spatial size of the cube, by default 128 (pix).
    coarsest_res_sensor : int, optional
        The coarsest resolution of the sensor, by default 60 (m).
    filter_by : str, optional
        The path to the geopackage to filter the locations, by default None.

    Returns
    -------
    dict
        A dictionary with the location name as key and the corresponding tile, epsg and geobox as values.
    """
    locs = pd.read_csv(locations_csv_path)
    grid = gpd.read_parquet(grid_parquet_path)

    ratio_for_geobox = coarsest_res_sensor // target_res_spatial

    locs["geometry"] = locs.apply(
        lambda x: Point((float(x.longitude), float(x.latitude))), axis=1
    )

    if filter_by:
        locs_gpd = gpd.GeoDataFrame(locs, geometry="geometry")
        locs_gpd.crs = "EPSG:4326"

        filter_by = gpd.read_file(filter_by)
        locs = locs[locs_gpd.intersects(filter_by.union_all())].reset_index(drop=True)

    locs_and_tiles = {}
    for i in range(len(locs)):
        pos = locs.loc[i]
        loc_name = f"{pos.SITE_ID}_{pos.geometry.y:.2f}_{pos.geometry.x:.2f}"

        grid_subset = grid[grid.intersects(pos.geometry)][
            ["tile", "utm_bounds", "epsg"]
        ]

        found = False
        for nr_tiles in range(len(grid_subset)):
            tile = grid_subset.tile.values[nr_tiles]
            epsg = grid_subset.epsg.values[nr_tiles]
            utm_bounds = grid_subset.utm_bounds.values[nr_tiles]
            utm_bounds = [
                float(value)
                for value in utm_bounds.replace("(", "").replace(")", "").split(",")
            ]
            bbox = BoundingBox(*utm_bounds)
            geobox_60m = GeoBox.from_bbox(bbox, crs=int(epsg), resolution=60)
            # transform the point to the grid crs
            pos_utm = transform(pos.geometry, get_transform(4326, int(epsg)))

            y_center = np.argmin(np.abs(geobox_60m.coordinates["y"].values - pos_utm.y))
            x_center = np.argmin(np.abs(geobox_60m.coordinates["x"].values - pos_utm.x))
            if (
                y_center > (spatial_size // 2) // ratio_for_geobox
                and x_center > (spatial_size // 2) // ratio_for_geobox
                and y_center
                < geobox_60m.shape[0] - (spatial_size // 2) // ratio_for_geobox
                and x_center
                < geobox_60m.shape[1] - (spatial_size // 2) // ratio_for_geobox
            ):
                min_y = y_center - (spatial_size // 2) // ratio_for_geobox
                min_x = x_center - (spatial_size // 2) // ratio_for_geobox

                request_box = geobox_60m[
                    min_y : min_y + spatial_size // ratio_for_geobox,
                    min_x : min_x + spatial_size // ratio_for_geobox,
                ]
                found = True
                if loc_name in locs_and_tiles:
                    print(f"Warning: {loc_name} already in locs_and_tiles", flush=True)
                locs_and_tiles[loc_name] = [tile, epsg, request_box]
                break

        if not found:
            if loc_name in locs_and_tiles:
                print(f"Warning: {loc_name} already in locs_and_tiles", flush=True)
            locs_and_tiles[loc_name] = ["not_found", "not_found", "not_found"]

    return locs_and_tiles


# utility
def run_with_multiprocessing(target_function: function, **func_kwargs):
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
        process = multiprocessing.Process(
            target=target_function, kwargs=func_kwargs, name=f"throwawayWorker"
        )

        process.start()
        process.join()
        if process.exitcode is None:
            print(f"Process Worker did not exit cleanly, terminating...")
            process.terminate()

        print(f"Process Worker completed with exit code {process.exitcode}")
        if process.exitcode == 0:
            break

    time.sleep(0.1)


# # utility
# def resample_s2(ds:xr.Dataset, upscale_factor, resampling):
#     return ds.rio.reproject(
#         ds.rio.crs,
#         shape=(int(ds.rio.width * upscale_factor), int(ds.rio.width * upscale_factor)),
#         resampling=resampling,
#     )


# # utility
# def add_s2_parts_resample(file_list, upscale_factor=0):
#     if upscale_factor > 0:
#         resampling = Resampling.nearest
#     if upscale_factor < 0:
#         resampling = Resampling.bilinear

#     # xds = xr.open_zarr(file_list[0])
#     # crs = CRS.from_wkt(xds.spatial_ref.attrs["spatial_ref"])
#     # xds.rio.write_crs(crs, inplace=True)

#     # if upscale_factor != 0:
#     #     xds = resample_s2(xds, upscale_factor, resampling)

#     xds_list = []

#     for _file in file_list:
#         xds_t = xr.open_zarr(_file)
#         crs = CRS.from_wkt(xds_t.spatial_ref.attrs["spatial_ref"])
#         xds_t.rio.write_crs(crs, inplace=True)

#         if upscale_factor != 0:
#             xds_t = resample_s2(xds_t, upscale_factor, resampling)

#         xds_list.append(xds_t)

#     return xr.merge(xds_list)


# utility
def harmonize_to_old(data:xr.Dataset, scale:bool=True) -> xr.Dataset:
    """
    Harmonize new Sentinel-2 data to the old baseline. Data after 25-01-2022 is clipped 
    to 1000 and then subtracted by 1000.
    From https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change
    adjusted to odc-stac, using different variables for each band.

    Parameters
    ----------
    data: xarray.DataArray
        A DataArray with four dimensions: time, band, y, x

    Returns
    -------
    harmonized: xarray.DataArray
        A DataArray with all values harmonized to the old
        processing baseline.
    """
    cutoff = datetime.datetime(2022, 1, 25)
    offset = 1000
    bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]

    to_process = list(set(bands) & set(data.keys()))
    no_change = data.drop_vars(to_process)
    old = data[to_process].sel(time=slice(cutoff))
    new_harmonized = data[to_process].sel(time=slice(cutoff, None)).clip(offset)
    new_harmonized -= offset

    new = xr.concat([old, new_harmonized], "time")
    if scale:
        new = new.where(new != 0)
        new = (new * 0.0001).astype("float32")
    else:
        new = new.astype("uint16")

    for variable in list(no_change.keys()):
        new[variable] = no_change[variable]
    return new

# utility
def drop_no_data_s2(cube:xr.Dataset, nodata_flag:int=0):
    """
    Dropping all-no-data time-steps, keeping int16 S2 Datasets in their dtype.
    Remove time slices where the mean of the B* bands is equal to the nodata_flag.

    Parameters
    ----------
    cube: xarray.Dataset
        A Dataset with three dimensions: time, y, x
    nodata_flag: int, optional
        The no-data flag to use, by default 0

    Returns
    -------
    xarray.Dataset
        A Dataset with all-no-data time-steps removed.
    """
    var_name = [var for var in cube.data_vars if var.startswith("B")][0]
    mean_over_time = cube[var_name].mean(dim=["x", "y"])
    return cube.isel(time=np.where(mean_over_time != nodata_flag)[0])

# utility
def __open_s2_multires_cube_to_20m(paths_10m:list[str], paths_20m:list[str], 
                                   paths_60m:list[str]) -> xr.Dataset:
    """
    Open the batched Sentinel-2 from S2_cube_part_in_native_res and 
    merge them into one Dataset with 20m target resolution.
    Resampling 10m bands to 20m using linear interpolation 
    and 60m bands to 20m using nearest neighbor.

    Parameters
    ----------
    paths_10m : list[str]
        List of paths to the 10m resolution cubes.
    paths_20m : list[str]
        List of paths to the 20m resolution cubes.
    paths_60m : list[str]
        List of paths to the 60m resolution cubes.

    Returns
    -------
    xarray.Dataset
        A Dataset with all the bands merged into one Dataset.
    """

    xds_s2_20m = xr.merge([xr.open_zarr(f).compute() for f in paths_20m])
    target_x = xds_s2_20m["x"]
    target_y = xds_s2_20m["y"]
    xds_s2_10m = xr.merge(
        [
            xr.open_zarr(f)
            .interp(x=target_x, y=target_y, method="linear")
            .round()
            .astype("uint16")
            .compute()
            for f in paths_10m
        ]
    )
    xds_s2_60m = xr.merge(
        [
            xr.open_zarr(f)
            .interp(
                x=target_x,
                y=target_y,
                method="nearest",
                kwargs={"fill_value": "extrapolate"},
            )
            .compute()
            for f in paths_60m
        ]
    )

    return xr.merge([xds_s2_10m, xds_s2_20m, xds_s2_60m])


# Combination
def combine_and_save(location:str) -> None:
    """
    find all parts for a location, combine them and save them to a single zipped zarr file.

    Parameters
    ----------

    location : str
        The location to process.

    Returns
    -------
    None

    """

    ESA_WC_ATTRS = {
        # 'wavelength' :  ,
        "dims": ["y", "x"],
        "flag_meanings": [
            "Tree cover",
            "Shrubland",
            "Grassland",
            "Cropland",
            "Built-up",
            "Bare / sparse vegetation",
            "Snow and ice",
            "Permanent water bodies",
            "Herbaceous wetland",
            "Mangroves",
            "Moss and lichen",
        ],
        "flag_values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
        # 'AREA_OR_POINT' :   ,
        # 'add_offset' :  ,
        #'dtype' :        'uint8',
        "long_name": "ESA WorldCover product 2021",
        "metadata": {
            "color_bar_name": "LC Class",
            "color_value_max": 100,
            "color_value_min": 10,
            "keywords": ["ESA WorldCover", "Classes"],
        },
        "name": "WorldCover21",
        # 'scale_factor' :  1,
        "sources": [
            "https://planetarycomputer.microsoft.com/api/stac/v1/collections/esa-worldcover"
        ],
        "units": "n.a.",
        #'_FillValue' :    0,
        "NoData": 0,
    }

    MODIS_DTYPES = {
        "250m_16_days_EVI": "int16",
        "250m_16_days_NDVI": "int16",
        "250m_16_days_VI_Quality": "uint16",
        "250m_16_days_MIR_reflectance": "int16",
        "250m_16_days_NIR_reflectance": "int16",
        "250m_16_days_red_reflectance": "int16",
        "250m_16_days_blue_reflectance": "int16",
        "250m_16_days_sun_zenith_angle": "int16",
        "250m_16_days_pixel_reliability": "int16",
        "250m_16_days_view_zenith_angle": "int16",
        "250m_16_days_relative_azimuth_angle": "int16",
    }
    MODIS_SPECS = {
        "250m_16_days_NDVI": {
            "dtype": "int16",
            "NoData": -3000,
            "data_scale_factor": 0.0001,
            "valid_range": (-2000, 10000),
            "orgiginal_spatial_resolution": 250,
            "resampling": "nearest",
        },
        "250m_16_days_EVI": {
            "dtype": "int16",
            "NoData": -3000,
            "data_scale_factor": 0.0001,
            "valid_range": (-2000, 10000),
            "orgiginal_spatial_resolution": 250,
            "resampling": "nearest",
        },
        "250m_16_days_VI_Quality": {
            "dtype": "uint16",
            "NoData": 65535,
            "valid_range": (0, 65534),
            "orgiginal_spatial_resolution": 250,
            "resampling": "nearest",
        },
        "250m_16_days_MIR_reflectance": {
            "dtype": "int16",
            "NoData": -1000,
            "data_scale_factor": 0.0001,
            "valid_range": (0, 10000),
            "orgiginal_spatial_resolution": 250,
            "resampling": "nearest",
        },
        "250m_16_days_NIR_reflectance": {
            "dtype": "int16",
            "NoData": -1000,
            "data_scale_factor": 0.0001,
            "valid_range": (0, 10000),
            "orgiginal_spatial_resolution": 250,
            "resampling": "nearest",
        },
        "250m_16_days_red_reflectance": {
            "dtype": "int16",
            "NoData": -1000,
            "data_scale_factor": 0.0001,
            "valid_range": (0, 10000),
            "orgiginal_spatial_resolution": 250,
            "resampling": "nearest",
        },
        "250m_16_days_blue_reflectance": {
            "dtype": "int16",
            "NoData": -1000,
            "data_scale_factor": 0.0001,
            "valid_range": (0, 10000),
            "orgiginal_spatial_resolution": 250,
            "resampling": "nearest",
        },
        "250m_16_days_sun_zenith_angle": {
            "dtype": "int16",
            "NoData": -10000,
            "data_scale_factor": 0.01,
            "valid_range": (0, 18000),
            "orgiginal_spatial_resolution": 250,
            "resampling": "nearest",
        },
        "250m_16_days_pixel_reliability": {
            "dtype": "int16",
            "NoData": -1,
            "valid_range": (0, 3),
            "orgiginal_spatial_resolution": 250,
            "resampling": "nearest",
        },
        "250m_16_days_view_zenith_angle": {
            "dtype": "int16",
            "NoData": -10000,
            "data_scale_factor": 0.01,
            "valid_range": (0, 18000),
            "orgiginal_spatial_resolution": 250,
            "resampling": "nearest",
        },
        "250m_16_days_relative_azimuth_angle": {
            "dtype": "int16",
            "NoData": -4000,
            "data_scale_factor": 0.01,
            "valid_range": (-18000, 18000),
            "orgiginal_spatial_resolution": 250,
            "resampling": "nearest",
        },
    }

    dev_path = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/_dev"
    final_path = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final"

    out_path = os.path.join(final_path, f"{location}_v0.zarr.zip")
    if os.path.exists(out_path):
        print(f"Skipping {location}", flush=True)
        return

    existing_cubes = os.listdir(dev_path)

    locs_dir = get_rs_product_tiles_for_locs(
        "4_TanDEM-X.csv",
        "sentinel-2-grid_LAND.parquet",
        filter_by="europa_dissolve.gpkg",
    )
    # location = 'PT-Mi2_38.48_-8.02'
    loc_meta = locs_dir[location]

    existing_cubes_old = [
        f for f in existing_cubes if "fluxnet_" + "_".join(location.split("_")[1:]) in f
    ]
    existing_cubes_old.sort()
    existing_cubes_old_s2 = [f for f in existing_cubes_old if "_S2_" in f]
    existing_cubes_old_modis = [f for f in existing_cubes_old if "MODIS" in f]

    existing_cubes_new = [f for f in existing_cubes if location in f]
    existing_cubes_new.sort()
    existing_cubes_new_s2_10m = [f for f in existing_cubes_new if "_10m" in f]
    existing_cubes_new_s2_20m = [f for f in existing_cubes_new if "_20m" in f]
    existing_cubes_new_s2_60m = [f for f in existing_cubes_new if "_60m" in f]
    existing_cubes_new_modis = [f for f in existing_cubes_new if "MODIS" in f]
    existing_cubes_new_wc = [f for f in existing_cubes_new if "ESA_WC" in f]

    existing_cubes_old_s2 = [os.path.join(dev_path, f) for f in existing_cubes_old_s2]
    existing_cubes_old_modis = [
        os.path.join(dev_path, f) for f in existing_cubes_old_modis
    ]
    existing_cubes_new_s2_10m = [
        os.path.join(dev_path, f) for f in existing_cubes_new_s2_10m
    ]
    existing_cubes_new_s2_20m = [
        os.path.join(dev_path, f) for f in existing_cubes_new_s2_20m
    ]
    existing_cubes_new_s2_60m = [
        os.path.join(dev_path, f) for f in existing_cubes_new_s2_60m
    ]
    existing_cubes_new_modis = [
        os.path.join(dev_path, f) for f in existing_cubes_new_modis
    ]
    existing_cubes_new_wc = [os.path.join(dev_path, f) for f in existing_cubes_new_wc]

    xds_s2 = __open_s2_multires_cube_to_20m(
        existing_cubes_new_s2_10m, existing_cubes_new_s2_20m, existing_cubes_new_s2_60m
    )

    # if len(existing_cubes_old_s2) > 0:
    #     time_old = [(time.split('_')[7][:4], time.split('_')[8][:4]) for time in existing_cubes_old_s2]
    #     timerange_old = [list(range(int(time[0]), int(time[1])+1)) for time in time_old]
    #     timerange_old = set([item for sublist in timerange_old for item in sublist])

    #     time_new = [(time.split('_')[8][:4], time.split('_')[9][:4]) for time in existing_cubes_new_s2_10m]
    #     timerange_new = [list(range(int(time[0]), int(time[1])+1)) for time in time_new]
    #     timerange_new = set([item for sublist in timerange_new for item in sublist])

    #     take_from_old_s2 = timerange_old - timerange_new
    #     paths_from_old = [path for path in existing_cubes_old_s2 if int(path.split('_')[7][:4]) in take_from_old_s2]

    #     old_s2_cubes = xr.merge([xr.open_zarr(os.path.join(dev_path, path)).compute() for path in paths_from_old]).isel(x=slice(0,-2), y=slice(0,-2))
    #     xds_s2 = xr.merge([old_s2_cubes, xds_s2])

    xds_s2 = drop_no_data_s2(xds_s2)

    xds_s2 = xds_s2.sortby("time")
    xds_s2 = harmonize_to_old(xds_s2)

    scl = xds_s2["SCL"]
    xds_s2 = xds_s2.where(xds_s2 != 0.0)
    xds_s2["SCL"] = scl

    xds_s2 = xds_s2.chunk({"time": -1, "x": 126 // 2, "y": 126 // 2})

    crs = CRS.from_wkt(
        xr.open_zarr(existing_cubes_new_s2_20m[0]).spatial_ref.attrs["spatial_ref"]
    )
    xds_s2.rio.write_crs(crs, inplace=True)
    # xds_s2 = xds_s2.drop_vars(["spatial_ref"])

    xds_modis = xr.merge([xr.open_zarr(i).compute() for i in existing_cubes_new_modis])
    crs = CRS.from_wkt(
        xr.open_zarr(existing_cubes_new_modis[0]).spatial_ref.attrs["spatial_ref"]
    )
    xds_modis.rio.write_crs(crs, inplace=True)
    # xds_modis = xds_modis.drop_vars(["spatial_ref"])

    if not xds_modis.rio.crs == xds_s2.rio.crs:
        target_x = np.arange(xds_modis.x.min().values, xds_modis.x.max().values, 20)
        target_y = np.arange(xds_modis.y.min().values, xds_modis.y.max().values, 20)
        xds_modis = xds_modis.interp(
            x=target_x,
            y=target_y,
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )

        # Step 2: Resample the source data to the target grid
        # Calculate the transform and dimensions of the reprojected raster
        transform, width, height = calculate_default_transform(
            crs,
            f"EPSG:{loc_meta[1]}",
            # TODO: check if this can handle the upsampling aswell
            len(xds_modis.x),
            len(xds_modis.y),
            *xds_modis.rio.bounds(),
        )
        # use transform and dimensions to create a new xarray dataset
        xds_modis = xds_modis.rio.reproject(
            f"EPSG:{loc_meta[1]}",
            transform=transform,
            shape=(height, width),
            resampling=Resampling.nearest,
        )

        target_x = xds_s2["x"]
        target_y = xds_s2["y"]

        # Step 2: Resample the source data to the target grid
        xds_modis = xds_modis.interp(
            x=target_x,
            y=target_y,
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )
        xds_modis = xds_modis.rio.clip_box(
            *loc_meta[2].boundingbox, crs=f"EPSG:{loc_meta[1]}"
        )

    # if len(existing_cubes_old_modis) > 0:
    #     time_old = [(time.split('_')[7][:4], time.split('_')[8][:4]) for time in existing_cubes_old_modis]
    #     timerange_old = [list(range(int(time[0]), int(time[1])+1)) for time in time_old]
    #     timerange_old = set([item for sublist in timerange_old for item in sublist])

    #     time_new = [(time.split('_')[8][:4], time.split('_')[9][:4]) for time in existing_cubes_new_modis]
    #     timerange_new = [list(range(int(time[0]), int(time[1])+1)) for time in time_new]
    #     timerange_new = set([item for sublist in timerange_new for item in sublist])

    #     take_from_old_modis = timerange_old - timerange_new

    #     paths_from_old = [path for path in existing_cubes_old_modis if int(path.split('_')[7][:4]) in take_from_old_modis]

    #     old_modis = xr.merge([xr.open_zarr(os.path.join(dev_path, path)).compute() for path in paths_from_old]).isel(x=slice(0,-2), y=slice(0,-2))
    #     xds_modis = xr.merge([old_modis, xds_modis])

    for k in MODIS_SPECS.keys():
        xds_modis[k] = xds_modis[k].fillna(MODIS_SPECS[k]["NoData"])

    xds_modis = xds_modis.rename({"time": "start_range"}).sortby("start_range")
    xds_modis = xds_modis.chunk({"start_range": -1, "x": 126 // 2, "y": 126 // 2})
    xds_modis = xds_modis.astype(MODIS_DTYPES)

    for k in MODIS_SPECS.keys():
        xds_modis[k].attrs = MODIS_SPECS[k]
        xds_modis[k].attrs["long_name"] = (f"16 Day Reflectance in band {k}",)
        xds_modis[k].attrs[
            "source"
        ] = "https://planetarycomputer.microsoft.com/api/stac/v1/modis-13Q1-061"
        xds_modis[k].attrs["units"] = ("n.a.",)
        xds_modis[k].attrs["metadata"] = {
            "color_bar_name": "gray",
            "color_value_max": 1.0,
            "color_value_min": 0.0,
            "keywords": ["MODIS", "Reflectances", "Indices"],
        }

    xds_wc = xr.open_zarr(existing_cubes_new_wc[0])
    crs = CRS.from_wkt(xds_wc.spatial_ref.attrs["spatial_ref"])
    xds_wc.rio.write_crs(crs, inplace=True)

    if not xds_wc.rio.crs == xds_s2.rio.crs:
        transform, width, height = calculate_default_transform(
            crs,
            f"EPSG:{loc_meta[1]}",
            len(xds_wc.longitude),
            len(xds_wc.latitude),
            *xds_wc.rio.bounds(),
        )
        xds_wc = xds_wc.rio.reproject(
            f"EPSG:{loc_meta[1]}",
            transform=transform,
            shape=(height, width),
            resampling=Resampling.nearest,
        )
        target_x = xds_s2["x"]
        target_y = xds_s2["y"]

        xds_wc = xds_wc.interp(x=target_x, y=target_y, method="nearest")
        xds_wc = xds_wc.rio.clip_box(
            *loc_meta[2].boundingbox, crs=f"EPSG:{loc_meta[1]}"
        )
    xds_wc = xds_wc.squeeze().drop_vars(["time"])  # , "spatial_ref"])
    xds_wc = xds_wc.rename_vars({"map": "esa_worldcover_2021"})
    xds_wc = xds_wc.chunk({"x": -1, "y": -1})
    xds_wc = xds_wc.astype("uint8")

    xds_wc["esa_worldcover_2021"].attrs = ESA_WC_ATTRS

    s2_specs = {
        "B02": {
            "S-2Acentralwavelength": 492.7,
            "S-2Abandwidth": 65,
            "S-2Bcentralwavelength": 492.3,
            "S-2Bbandwidth": 65,
            "reference_radiance": 128,
            "SNR_at_Lref": 154,
        },
        "B03": {
            "S-2Acentralwavelength": 559.8,
            "S-2Abandwidth": 35,
            "S-2Bcentralwavelength": 558.9,
            "S-2Bbandwidth": 35,
            "reference_radiance": 128,
            "SNR_at_Lref": 168,
        },
        "B04": {
            "S-2Acentralwavelength": 664.6,
            "S-2Abandwidth": 30,
            "S-2Bcentralwavelength": 664.9,
            "S-2Bbandwidth": 31,
            "reference_radiance": 108,
            "SNR_at_Lref": 142,
        },
        "B08": {
            "S-2Acentralwavelength": 832.8,
            "S-2Abandwidth": 105,
            "S-2Bcentralwavelength": 832.9,
            "S-2Bbandwidth": 104,
            "reference_radiance": 103,
            "SNR_at_Lref": 174,
        },
        "B05": {
            "S-2Acentralwavelength": 704.1,
            "S-2Abandwidth": 14,
            "S-2Bcentralwavelength": 703.8,
            "S-2Bbandwidth": 15,
            "reference_radiance": 74.5,
            "SNR_at_Lref": 117,
        },
        "B06": {
            "S-2Acentralwavelength": 740.5,
            "S-2Abandwidth": 14,
            "S-2Bcentralwavelength": 739.1,
            "S-2Bbandwidth": 13,
            "reference_radiance": 68,
            "SNR_at_Lref": 89,
        },
        "B07": {
            "S-2Acentralwavelength": 782.8,
            "S-2Abandwidth": 19,
            "S-2Bcentralwavelength": 779.7,
            "S-2Bbandwidth": 19,
            "reference_radiance": 67,
            "SNR_at_Lref": 105,
        },
        "B8A": {
            "S-2Acentralwavelength": 864.7,
            "S-2Abandwidth": 21,
            "S-2Bcentralwavelength": 864.0,
            "S-2Bbandwidth": 21,
            "reference_radiance": 52.5,
            "SNR_at_Lref": 72,
        },
        "B11": {
            "S-2Acentralwavelength": 1613.7,
            "S-2Abandwidth": 90,
            "S-2Bcentralwavelength": 1610.4,
            "S-2Bbandwidth": 94,
            "reference_radiance": 4,
            "SNR_at_Lref": 100,
        },
        "B12": {
            "S-2Acentralwavelength": 2202.4,
            "S-2Abandwidth": 174,
            "S-2Bcentralwavelength": 2185.7,
            "S-2Bbandwidth": 184,
            "reference_radiance": 1.5,
            "SNR_at_Lref": 100,
        },
        "B01": {
            "S-2Acentralwavelength": 442.7,
            "S-2Abandwidth": 20,
            "S-2Bcentralwavelength": 442.3,
            "S-2Bbandwidth": 20,
            "reference_radiance": 129,
            "SNR_at_Lref": 129,
        },
        "B09": {
            "S-2Acentralwavelength": 945.1,
            "S-2Abandwidth": 19,
            "S-2Bcentralwavelength": 943.2,
            "S-2Bbandwidth": 20,
            "reference_radiance": 9,
            "SNR_at_Lref": 114,
        },
        "B10": {
            "S-2Acentralwavelength": 1373.5,
            "S-2Abandwidth": 29,
            "S-2Bcentralwavelength": 1376.9,
            "S-2Bbandwidth": 29,
            "reference_radiance": 6,
            "SNR_at_Lref": 50,
        },
    }

    for k in xds_s2.data_vars.keys():
        if k == "SCL":
            continue
        xds_s2[k].attrs = {
            "long_name": f"Reflectance in band {k}",
            "metadata": {
                "color_bar_name": "gray",
                "color_value_max": 1.0,
                "color_value_min": 0.0,
                "keywords": ["Sentinel-2", "Reflectances"],
            },
            "centralwavelength": {
                "S-2A": s2_specs[k]["S-2Acentralwavelength"],
                "S-2B": s2_specs[k]["S-2Bcentralwavelength"],
            },
            "bandwidth": {
                "S-2A": s2_specs[k]["S-2Abandwidth"],
                "S-2B": s2_specs[k]["S-2Bbandwidth"],
            },
            "dtype": "float32",
            "NoData": "nan",
            "source": "https://planetarycomputer.microsoft.com/api/stac/v1/sentinel-2-l2a",
        }

    xds_s2["SCL"] = xds_s2["SCL"].astype("uint8")
    xds_s2["SCL"].attrs = {
        "long_name": f"Scene classification mask",
        "dtype": "uint8",
        "NoData": 0,
        "units": "n.a.",
        "flag_meanings": [
            "Saturated / Defective",
            "Dark Area Pixels",
            "Cloud Shadows",
            "Vegetation",
            "Bare Soils",
            "Water",
            "Clouds low probability / Unclassified",
            "Clouds medium probability",
            "Clouds high probability",
            "Cirrus",
            "Snow / Ice",
        ],
        "flag_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    }

    xds_s2 = xds_s2.drop_vars(["spatial_ref"])
    xds_modis = xds_modis.drop_vars(["spatial_ref"])
    xds_wc = xds_wc.drop_vars(["spatial_ref"])

    xds = xr.merge([xds_s2, xds_modis, xds_wc])

    xds.attrs = {
        "Name": "FluxSitesMCDatasetv0",
        "EPSG": int(loc_meta[1]),
        "bbox": xds.rio.bounds(),
        "Creation Time": datetime.datetime.now().isoformat(),
    }

    store = zarr.ZipStore(out_path, mode="w")

    xds.to_zarr(out_path, mode="w")
    store.close()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        min_flux = int(sys.argv[1])
        max_flux = int(sys.argv[2])
        # year = int(sys.argv[3])
    else:
        min_flux = 153
        max_flux = 154

    print(f"Processing {min_flux} to {max_flux}", flush=True)
    dev_path = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/_dev"
    existing_cubes = os.listdir(dev_path)

    time_span_s2 = (2015, 2026)
    time_span_modis = (2014, 2026)

    sel_bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "SCL",
    ]
    locs_dir = get_rs_product_tiles_for_locs(
        "4_TanDEM-X.csv",
        "sentinel-2-grid_LAND.parquet",
        filter_by="europa_dissolve.gpkg",
    )
    locs_dir_s2_boxes = copy.deepcopy(locs_dir)

    # existing_dict, unfinished = __filter_for_existing_cubes(
    #     existing_cubes, sensor="S2", locations=list(locs_dir.keys())
    # )
    # locs_dir = __filter_years_to_process(
    #     locs_dir, existing_dict, year_range_process=(2015, 2026)
    # )

    for i, location in enumerate(islice(locs_dir.keys(), min_flux, max_flux)):
        timerange = time_span_s2
        print(
            i,
            location,
            locs_dir[location][0],
            f"{timerange[0]}-01-01",
            f"{timerange[-1]}-12-31",
            flush=True,
        )
        query = request_data_by_tile(
            locs_dir[location][0],
            f"{timerange[0]}-01-01",
            f"{timerange[-1]}-12-31",
            save_dir=dev_path,
        )  # , previous_queries=existing_queries_s2)

        run_with_multiprocessing(
            S2_cube_part_in_native_res,
            items=query,
            out_path=os.path.join(
                dev_path,
                f"fluxnet_{location}_S2_v0_{timerange[0]}-01-01_{timerange[-1]}-12-31",
            ),
            request_geobox=locs_dir[location][2],
            sel_bands=sel_bands,
            asset_bin_size=50,
        )

    print("Done with S2", flush=True)
    ### modis part
    modis_grid = gpd.read_parquet("modis-13Q1-061_tiles_dissolved.parquet")
    modis_bands = [
        "250m_16_days_EVI",
        "250m_16_days_NDVI",
        "250m_16_days_VI_Quality",
        "250m_16_days_MIR_reflectance",
        "250m_16_days_NIR_reflectance",
        "250m_16_days_red_reflectance",
        "250m_16_days_blue_reflectance",
        "250m_16_days_sun_zenith_angle",
        "250m_16_days_pixel_reliability",
        "250m_16_days_view_zenith_angle",
        "250m_16_days_relative_azimuth_angle",
    ]

    # existing_dict, unfinished_modis = __filter_for_existing_cubes(
    #     existing_cubes, sensor="MODIS", locations=list(locs_dir_s2_boxes.keys())
    # )
    # locs_dir_s2_boxes = __filter_years_to_process(locs_dir_s2_boxes, existing_dict)

    for i, location in enumerate(islice(locs_dir_s2_boxes.keys(), min_flux, max_flux)):
        transformed_box = transform(
            box(*locs_dir_s2_boxes[location][2].boundingbox),
            get_transform(locs_dir_s2_boxes[location][1], 4326),
        )
        # todo: the folowwing could lead to errors if multiple modis tiles are found intersecting the aoi!!!
        modis_tile = modis_grid[
            modis_grid.geometry.contains(transformed_box)
        ].index.values[0]
        timerange = time_span_modis
        print(
            i,
            location,
            modis_tile,
            f"{timerange[0]}-01-01",
            f"{timerange[-1]}-12-31",
            flush=True,
        )
        query = request_data_by_tile(
            modis_tile,
            f"{timerange[0]}-01-01",
            f"{timerange[-1]}-12-31",
            collection="modis-13Q1-061",
            save_dir=dev_path,
        )  # , previous_queries=existing_queries_s2)

        run_with_multiprocessing(
            MODIS_cube_part,
            items=query,
            out_path=os.path.join(
                dev_path,
                f"fluxnet_{location}_MODIS_v0_{timerange[0]}-01-01_{timerange[-1]}-12-31",
            ),
            request_geobox=locs_dir[location][2],
            sel_bands=modis_bands,
            asset_bin_size=50,
        )

    print("Done with MODIS", flush=True)

    esa_wc_grid = gpd.read_parquet("esa-worldcover.parquet")
    esa_wc_grid = esa_wc_grid.where(
        esa_wc_grid["esa_worldcover:product_version"] == "2.0.0"
    ).dropna(how="all")

    for i, location in enumerate(islice(locs_dir.keys(), min_flux, max_flux)):
        print(i, location, "ESA WC", flush=True)
        transformed_box = transform(
            box(*locs_dir[location][2].boundingbox),
            get_transform(locs_dir[location][1], 4326),
        )
        subset_match = esa_wc_grid.where(
            esa_wc_grid.intersects(transformed_box) == True
        ).dropna(how="all")
        items = to_item_collection(subset_match)

        for _item in range(len(items)):
            items[_item].properties["proj:epsg"] = int(
                items[_item].properties["proj:epsg"]
            )

        out_wc = os.path.join(dev_path, f"fluxnet_{location}_ESA_WC_v0_2021.zarr.zip")
        if not os.path.exists(out_wc):
            bounds = list(locs_dir[location][2].boundingbox)
            crs = locs_dir[location][2].crs

            box_20m = GeoBox.from_bbox(bounds, crs, resolution=20)
            run_with_multiprocessing(
                save_stac_to_zarr_zip,
                items=items,
                out_path=out_wc,
                bands=["map"],
                dtype="uint8",
                geobox=box_20m,
            )
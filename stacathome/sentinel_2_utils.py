import datetime
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import xarray as xr
import zarr
from odc import stac
from odc.geo.geobox import GeoBox
from pyproj import CRS
from rasterio.errors import RasterioIOError, WarpOperationError
from shapely import box, transform

from .asset_specs import get_attributes, get_band_attributes_s2
from .request import request_data_by_tile
from .utils import filter_items_to_data_coverage, get_time_binning, get_transform, run_with_multiprocessing


BASE_ATTRS = {
    'Name': 'WeGenS2Dataset',
    'Version': '0.0',
}


def get_s2_multires_zip(
    bucket: gpd.GeoDataFrame,
    time_bins: int | tuple[int] | tuple[str],
    work_dir: str,
    bands: list[str] = None,
    verbose: bool = False,
):
    time_bins = get_time_binning(time_bins)

    if bands is None:
        bands = list(get_attributes('sentinel-2-l2a')['data_attrs']['Band'])
        bands.remove('B10')

    tile, number = bucket.tile.values[0].split('_')
    number = number.zfill(3)

    data_path = os.path.join(work_dir, tile, number)
    query_path = os.path.join(work_dir, tile, 'sentinel-2-l2a_queries')

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(query_path, exist_ok=True)

    items = request_data_by_tile(
        tile_id=tile,
        start_time=min(time_bins).strftime('%Y-%m-%d'),
        end_time=max(time_bins).strftime('%Y-%m-%d'),
        save_dir=query_path,
    )
    bounds = list(bucket.utm_bounds.values[0])
    crs = bucket.epsg.values[0]
    transformed_box = transform(box(*bounds), get_transform(str(crs), 4326))
    items = filter_items_to_data_coverage(items, transformed_box, sensor='sentinel-2-l2a')

    for t_bin in range(len(time_bins) - 1):
        if verbose:
            print(time_bins[t_bin], time_bins[t_bin + 1])
        items_in_timestamp = [
            item
            for item in items
            if time_bins[t_bin] <= pd.Timestamp(item.properties["datetime"][:-1]) <= time_bins[t_bin + 1]
        ]

        t_bin_s = time_bins[t_bin].strftime('%Y-%m-%d')
        t_bin_e = time_bins[t_bin + 1].strftime('%Y-%m-%d')
        if verbose and len(items_in_timestamp) == 0:
            print(f'Skipping {t_bin_s} to {t_bin_e}, no items found', flush=True)
            continue

        run_with_multiprocessing(
            load_and_save_s2_multires_zip,
            data_path=data_path,
            items_in_timestamp=items_in_timestamp,
            bands=bands,
            bounds=bounds,
            t_bin_s=t_bin_s,
            t_bin_e=t_bin_e,
            crs=crs,
            tile=tile,
            number=number,
            verbose=verbose,
        )


def load_and_save_s2_multires_zip(
    data_path, items_in_timestamp, bands, bounds, t_bin_s, t_bin_e, crs, tile, number, verbose
):
    s2_attributes = get_attributes('sentinel-2-l2a')['data_attrs']
    _bands_10m = set(s2_attributes['Band'].where(s2_attributes['Spatial Resolution'] == 10).dropna().to_list())
    _bands_20m = set(s2_attributes['Band'].where(s2_attributes['Spatial Resolution'] == 20).dropna().to_list())
    _bands_60m = set(s2_attributes['Band'].where(s2_attributes['Spatial Resolution'] == 60).dropna().to_list())
    _s2_dytpes = dict(zip(s2_attributes["Band"], s2_attributes["Data Type"]))

    box_10m = GeoBox.from_bbox(bounds, CRS.from_epsg(crs), resolution=10)
    box_20m = GeoBox.from_bbox(bounds, CRS.from_epsg(crs), resolution=20)
    box_60m = GeoBox.from_bbox(bounds, CRS.from_epsg(crs), resolution=60)
    sel_bands_s = set(bands)
    bands_10m = list(_bands_10m & sel_bands_s)
    bands_20m = list(_bands_20m & sel_bands_s)
    bands_60m = list(_bands_60m & sel_bands_s)
    # split here for memleak debugging
    multires_cube = {}

    out_path = os.path.join(data_path, f"{tile}_{number}_{t_bin_s}_{t_bin_e}_S2_v0.zarr.zip")

    for bands, gbox in zip([bands_10m, bands_20m, bands_60m], [box_10m, box_20m, box_60m]):
        if len(bands) == 0:
            continue
        if verbose:
            print(f"Loading {bands} for {t_bin_s} to {t_bin_e}", flush=True)

        start_time = time.time()
        parameters = {
            "items": items_in_timestamp,
            "patch_url": pc.sign,
            "bands": bands,
            "dtype": _s2_dytpes,
            "chunks": {"time": 2, "x": -1, "y": -1},
            "groupby": "solar_day",
            "resampling": "nearest",
            "fail_on_error": True,
            "geobox": gbox,
        }
        for _ in range(5):
            try:
                multires_cube[f"S2_{int(gbox.resolution.x)}m"] = stac.load(**parameters).compute()
                if verbose:
                    print(  # TODO: to be replaced with logging
                        f"Time taken for {t_bin_s} to {t_bin_e} for {out_path}: {time.time() - start_time:.0f} s",
                        flush=True,
                    )
                break
            except (WarpOperationError, RasterioIOError) as e:
                print(f"Error creating {bands} for {t_bin_s} to {t_bin_e}: {e}", flush=True)
                time.sleep(5)

    if verbose and len(multires_cube) == 0:
        print(f"Skipping {t_bin_s} to {t_bin_e}, no bands found", flush=True)
        return

    for spat_res in [60, 20, 10]:
        if f'S2_{spat_res}m' in multires_cube:
            use = f'S2_{spat_res}m'
            break
    mean_over_time = multires_cube[use][list(multires_cube['S2_60m'].data_vars.keys())[0]].mean(dim=['x', 'y'])

    for spat_res in [60, 20, 10]:
        cube_name = f'S2_{spat_res}m'
        if cube_name in multires_cube:
            multires_cube[cube_name] = multires_cube[cube_name].isel(time=np.where(mean_over_time != 0)[0])
            if cube_name in ['S2_20m', 'S2_60m']:
                multires_cube[cube_name] = multires_cube[cube_name].rename({'x': f'x{spat_res}', 'y': f'y{spat_res}'})

    multires_cube = xr.merge(multires_cube.values())
    multires_cube = harmonize_to_old(multires_cube, scale=True)

    cube_attrs = {
        'EPSG': crs,
        'UTM Tile': tile,
        'Bucket': tile + '_' + number,
        'Creation Time': datetime.datetime.now().isoformat(),
    }
    multires_cube.attrs = {**BASE_ATTRS, **cube_attrs}

    # s2_attributes = get_attributes('sentinel-2-l2a')['data_attrs']
    attributes = get_attributes('sentinel-2-l2a')
    data_atts = attributes.pop("data_attrs")
    band_attrs = get_band_attributes_s2(data_atts, multires_cube.data_vars.keys())
    for band in band_attrs.keys():
        multires_cube[band].attrs = band_attrs[band]
    multires_cube['SCL'] = attributes['SCL']

    multires_cube.chunk({"time": 2, "x": -1, "y": -1, "x20": -1, "y20": -1, "x60": -1, "y60": -1})
    store = zarr.ZipStore(out_path, mode="w")
    multires_cube.to_zarr(store, mode="w")
    store.close()
    return


def harmonize_to_old(data: xr.Dataset, scale: bool = True) -> xr.Dataset:
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

    bands = list(get_attributes('sentinel-2-l2a')['data_attrs']["Band"])
    if "SCL" in bands:
        bands.remove("SCL")

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


def drop_no_data_s2(cube, nodata_flag=0):
    var_name = [var for var in cube.data_vars if var.startswith('B')][0]
    mean_over_time = cube[var_name].mean(dim=['x', 'y'])
    return cube.isel(time=np.where(mean_over_time != nodata_flag)[0])


def remove_scl_invalid_pixels(xr: xr.Dataset, valid_values: list[int] = None):
    if valid_values is None:
        valid_values = [2, 4, 5, 6, 7, 11]

    scl_masks = xr.where(xr.SCL.isin(valid_values), 1, 0)
    return xr.where(scl_masks == 1, xr, 0)

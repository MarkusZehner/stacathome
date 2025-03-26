import datetime
import os
import time
# from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import xarray as xr
import zarr
from odc import stac
from odc.geo.geobox import GeoBox
# from odc.geo.xr import xr_zeros
# from odc.stac.model import _convert_to_solar_time


from pyproj import CRS
from rasterio.errors import RasterioIOError, WarpOperationError
from shapely import box, transform
import zipfile
# import warnings


from .asset_specs import get_attributes, get_band_attributes_s2, base_attrs
from .request import request_data_by_tile
from .utils import filter_items_to_data_coverage, get_transform, run_with_multiprocessing, run_with_multiprocessing_and_return, compute_scale_and_offset


# def create_empty_multires_cube(data_path, items, time_chunk_size,
#                                bands, bounds, t_bin_s, t_bin_e, crs, tile, number, overwrite=False):
#     out_path = os.path.join(data_path, f"{tile}_{number}_{t_bin_s}_{t_bin_e}_S2_v0.zarr")
#     if os.path.exists(out_path) and not overwrite:
#         warnings.warn(f"Cube {out_path} already exists, check if the specs are correct for your processing or set overwrite=True")
#         return out_path

#     lons, _ = zip(*items[0].geometry['coordinates'][0])
#     mid_lon = np.mean(lons)
#     part_convert_solar_day = partial(_convert_to_solar_time, longitude=mid_lon)

#     timeax = pd.to_datetime(
#         np.array(
#             [part_convert_solar_day(datetime.datetime.fromisoformat(item.properties['datetime'].rstrip('Z')))
#              for item in items],
#             dtype='datetime64[m]')
#     )
#     timeax_grouped = timeax.to_series().groupby(timeax.date).mean()
#     timeax_solarday = timeax_grouped.values.astype('datetime64[ns]')

#     # timeax = np.array([np.datetime64(item.properties['datetime'][:-1]) for item in items])
#     # timeax = timeax.astype('datetime64[ns]')
#     chunking = {'time': time_chunk_size, 'x': -1, 'y': -1}

#     s2_attributes = get_attributes('sentinel-2-l2a')['data_attrs']
#     _bands_10m = set(s2_attributes['Band'].where(s2_attributes['Spatial Resolution'] == 10).dropna().to_list())
#     _bands_20m = set(s2_attributes['Band'].where(s2_attributes['Spatial Resolution'] == 20).dropna().to_list())
#     _bands_60m = set(s2_attributes['Band'].where(s2_attributes['Spatial Resolution'] == 60).dropna().to_list())
#     _s2_dytpes = dict(zip(s2_attributes["Band"], s2_attributes["Data Type"]))

#     box_10m = GeoBox.from_bbox(bounds, CRS.from_epsg(crs), resolution=10)
#     box_20m = GeoBox.from_bbox(bounds, CRS.from_epsg(crs), resolution=20)
#     box_60m = GeoBox.from_bbox(bounds, CRS.from_epsg(crs), resolution=60)
#     sel_bands_s = set(bands)
#     bands_10m = list(_bands_10m & sel_bands_s)
#     bands_20m = list(_bands_20m & sel_bands_s)
#     bands_60m = list(_bands_60m & sel_bands_s)

#     empty_mc = {}
#     for bands, gbox in zip([bands_10m, bands_20m, bands_60m], [box_10m, box_20m, box_60m]):
#         if len(bands) == 0:
#             continue
#         xr_empty = xr_zeros(
#             gbox,
#             dtype=_s2_dytpes[bands[0]],
#             time=timeax_solarday,
#         )
#         xr_empty = xr_empty.to_dataset(name=bands[0])
#         for band in bands:
#             xr_empty[band] = xr_zeros(
#                 gbox,
#                 time=timeax_solarday,
#                 dtype=_s2_dytpes[band]
#             )
#             xr_empty[band].encoding["_FillValue"] = 0
#         xr_empty = xr_empty.chunk(chunking)
#         if int(gbox.resolution.x) in [20, 60]:
#             xr_empty = xr_empty.rename({'x': f'x{int(gbox.resolution.x)}',
#                                         'y': f'y{int(gbox.resolution.x)}'})

#         empty_mc[f"S2_{int(gbox.resolution.x)}m"] = xr_empty

#     empty_mc = xr.merge(empty_mc.values())

#     cube_attrs = {
#         'EPSG': crs,
#         'UTM Tile': tile,
#         'Bucket': tile + '_' + number,
#         'Creation Time': datetime.datetime.now().isoformat(),
#     }
#     empty_mc.attrs = {**base_attrs(), **cube_attrs}
#     # store = zarr.ZipStore(out_path,
#     #                       mode="x" if not overwrite else "w",
#     #                       compression=zipfile.ZIP_BZIP2)
#     empty_mc.to_zarr(out_path, mode="w-" if not overwrite else "w",
#                      consolidated=True, compute=False, write_empty_chunks=False)
#     # zarr.consolidate_metadata(store)
#     # store.close()
#     return out_path


def combine_loaded_cubes(
    path_list,
    out_path,
    bucket,
    remove_parts=True
):
    tile, number = bucket.tile.split('_')
    number = number.zfill(3)
    out_path = os.path.join(out_path, f'{tile}_{number}_combine.zarr.zip')

    if os.path.exists(out_path):
        return

    run_with_multiprocessing(
        do_combine,
        path_list=path_list,
        out_path=out_path,
        remove_parts=remove_parts
    )


def do_combine(path_list, out_path, remove_parts=True):
    cubes = [xr.open_zarr(i) for i in path_list]
    cube = xr.concat(cubes, dim='time')
    cube = cube.chunk({"time": 500, "x": 50, "y": 50, "x20": 50, "y20": 50})

    for i in cube.data_vars.keys():
        if i not in ['SCL', 'spatial_ref']:
            cube[i] = cube[i].astype("float32")
            cube[i].encoding = {"dtype": "uint16",
                                "scale_factor": compute_scale_and_offset(cube[i].values),
                                "add_offset": 0.0,
                                "_FillValue": 65535}
        elif i == 'SCL':
            cube[i] = cube[i].astype("uint8")

    store = zarr.ZipStore(out_path, mode="x", compression=zipfile.ZIP_BZIP2)
    cube.to_zarr(store, mode="w-", consolidated=True)
    store.close()

    if remove_parts:
        for i in path_list:
            os.remove(i)


def get_s2_multires_zip(
    bucket: gpd.GeoDataFrame,
    time_bins: int | tuple[int] | tuple[str],
    work_dir: str,
    bands: list[str] = None,
    # single_file=False,
    # single_file_time_chunk=50,
    # overwrite: bool = False,
    verbose: bool = False,
):
    # time_bins = get_time_binning(time_bins)

    if bands is None:
        bands = list(get_attributes('sentinel-2-l2a')['data_attrs']['Band'])
        if 'B10' in bands:
            bands.remove('B10')

    tile, number = bucket.tile.split('_')
    number = number.zfill(3)

    data_path = os.path.join(work_dir, tile, number)
    query_path = os.path.join(work_dir, tile, 'sentinel-2-l2a_queries')

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(query_path, exist_ok=True)

    items = request_data_by_tile(
        tile_id=tile,
        time_range=(min(time_bins).strftime('%Y-%m-%d') + '/' + max(time_bins).strftime('%Y-%m-%d')),
        save_dir=query_path,
    )
    bounds = list(bucket.utm_bounds)
    crs = bucket.epsg
    transformed_box = transform(box(*bounds), get_transform(str(crs), 4326))
    items = filter_items_to_data_coverage(items, transformed_box, sensor='sentinel-2-l2a')

    # cube_path = None
    # if single_file:
    #     # create empty zarr.zip file
    #     # store = zarr.ZipStore(out_path, mode="x", compression=zipfile.ZIP_BZIP2)
    #     # store.close()
    #     # single_file = store
    #     print('creating single file')
    #     cube_path = create_empty_multires_cube(
    #         data_path=data_path,
    #         items=items,
    #         time_chunk_size=single_file_time_chunk,
    #         bands=bands,
    #         bounds=bounds,
    #         t_bin_s=min(time_bins).strftime('%Y-%m-%d'),
    #         t_bin_e=max(time_bins).strftime('%Y-%m-%d'),
    #         crs=crs,
    #         tile=tile,
    #         number=number,
    #         overwrite=overwrite,
    #     )

    cube_paths = []
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
        if len(items_in_timestamp) == 0:
            if verbose:
                print(f'Skipping {t_bin_s} to {t_bin_e}, no items found', flush=True)
            continue

        c_name = run_with_multiprocessing_and_return(
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
            # single_file_path=cube_path,
            verbose=verbose,
        )
        if isinstance(c_name, str):
            cube_paths.append(c_name)

    return cube_paths


def load_and_save_s2_multires_zip(
    data_path, items_in_timestamp, bands, bounds, t_bin_s, t_bin_e, crs, tile, number,
    # single_file_path=None,
    verbose=False,
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

    # if single_file_path is not None:
    #     out_path = single_file_path
    # else:
    out_path = os.path.join(data_path, f"{tile}_{number}_{t_bin_s}_{t_bin_e}_S2_v0.zarr.zip")

    if os.path.exists(out_path):  # and not single_file_path:
        if verbose:
            print(f"Skipping {t_bin_s} to {t_bin_e}, already exists", flush=True)
        return out_path

    for bands, gbox in zip([bands_10m, bands_20m, bands_60m], [box_10m, box_20m, box_60m]):
        if len(bands) == 0:
            continue
        if verbose:
            print(f"Loading {bands} for {t_bin_s} to {t_bin_e}", flush=True)

        start_time = time.time()
        parameters = {
            "collection": ["sentinel-2-l2a"],
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
    mean_over_time = multires_cube[use][list(multires_cube[use].data_vars.keys())[0]].mean(dim=['x', 'y'])

    chunking = {"time": 2}
    for spat_res in [60, 20, 10]:
        cube_name = f'S2_{spat_res}m'
        if cube_name in multires_cube:
            multires_cube[cube_name] = multires_cube[cube_name].isel(time=np.where(mean_over_time != 0)[0])
            if cube_name in ['S2_20m', 'S2_60m']:
                chunking[f'x{spat_res}'] = -1
                chunking[f'y{spat_res}'] = -1
                multires_cube[cube_name] = multires_cube[cube_name].rename({'x': f'x{spat_res}', 'y': f'y{spat_res}'})
            else:
                chunking['x'] = -1
                chunking['y'] = -1

    multires_cube = xr.merge(multires_cube.values())
    multires_cube = harmonize_to_old(multires_cube, scale=True)

    cube_attrs = {
        'EPSG': crs,
        'UTM Tile': tile,
        'Bucket': tile + '_' + number,
        'Creation Time': datetime.datetime.now().isoformat(),
    }
    multires_cube.attrs = {**base_attrs(), **cube_attrs}

    attributes = get_attributes('sentinel-2-l2a')
    data_atts = attributes.pop("data_attrs")
    band_attrs = get_band_attributes_s2(data_atts, multires_cube.data_vars.keys())
    for band in band_attrs.keys():
        multires_cube[band].attrs = band_attrs[band]
    multires_cube['SCL'].attrs = attributes['SCL']

    multires_cube = multires_cube.chunk(chunking)

    for i in multires_cube.data_vars.keys():
        if i not in ['SCL', 'spatial_ref']:
            multires_cube[i] = multires_cube[i].astype("float32")
            multires_cube[i].encoding = {"dtype": "uint16",
                                         "scale_factor": compute_scale_and_offset(multires_cube[i].values),
                                         "add_offset": 0.0,
                                         "_FillValue": 65535}
        elif i == 'SCL':
            multires_cube[i] = multires_cube[i].astype("uint8")
            # multires_cube[i].encoding = {"dtype": "uint8",
            #                             #  "scale_factor": 1,
            #                             #  "add_offset": 0,
            #                              "_FillValue": 255}

    # if single_file_path is None:
    store = zarr.ZipStore(out_path, mode="x", compression=zipfile.ZIP_BZIP2)
    multires_cube.to_zarr(store, mode="w-", consolidated=True)
    store.close()
    # else:
    #     multires_cube['time'] = xr.open_zarr(single_file_path)['time'].sel(time=slice(pd.Timestamp(items_in_timestamp[0].properties['datetime'][:-1]).strftime('%Y-%m-%d'),
    #                                                                                   pd.Timestamp(items_in_timestamp[-1].properties['datetime'][:-1]).strftime('%Y-%m-%d'))).values
    #     # store = zarr.ZipStore(out_path, mode="w", compression=zipfile.ZIP_BZIP2)
    #     multires_cube = multires_cube.drop_vars(['y', 'x', 'y20', 'x20', 'spatial_ref'])
    #     multires_cube.to_zarr(out_path, consolidated=True,
    #                           region='auto')
    #     # store.close()
    return out_path


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

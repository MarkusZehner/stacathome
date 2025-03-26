import os
import datetime
import time
import zarr
import zipfile
from shapely import box, buffer, transform
from pystac import Item

from odc.geo.geobox import GeoBox
from pyproj import CRS
from odc import stac
import planetary_computer as pc
from rasterio.errors import RasterioIOError, WarpOperationError

from stacathome.utils import get_transform, run_with_multiprocessing, run_with_multiprocessing_and_return
from stacathome.request import request_data_by_bbox
from stacathome.asset_specs import get_attributes, base_attrs


def get_esa_wc(bucket, time_range, work_dir, verbose=False):
    """
    Request ESA World Cover data for a given time range and save it to the working directory.

    Parameters
    ----------
    bucket : str
        The bucket to download the data from.
    time : int
        2020 or 2021 for the two maps available (to date) to download.
    work_dir : str
        The working directory to save the data to.
    verbose : bool
        Whether to print verbose output.
    """
    # get the items
    assert time_range in [2020, 2021]

    tile, number = bucket.tile.split('_')
    number = number.zfill(3)

    data_path = os.path.join(work_dir, 'esa_wc_data')
    query_path = os.path.join(work_dir, 'esa_wc_queries')
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(query_path, exist_ok=True)

    bounds = list(bucket.utm_bounds)
    crs = bucket.epsg
    distance_in_m = 40
    buffered_box = box(*buffer(box(*bounds), distance_in_m / 2).bounds)
    buffered_transformed_box = transform(buffered_box,
                                         get_transform(str(crs), 4326))

    items = request_data_by_bbox(
        bbox=buffered_transformed_box,
        time_range=str(time_range),
        collection='esa-worldcover',
        save_dir=query_path,
    )
    # download the items
    if len(items) > 0:
        c_name = run_with_multiprocessing_and_return(
            load_and_save_esa_wc_zip,
            data_path=data_path,
            items_in_timestamp=items,
            bands=['map'],
            buffered_bounds=buffered_box.bounds,
            bounds=bounds,
            time_range=time_range,
            crs=crs,
            tile=tile,
            number=number,
            verbose=verbose,
        )
    return c_name


def load_and_save_esa_wc_zip(
    data_path: str,
    items_in_timestamp: list[Item],
    bands: list[str],
    buffered_bounds,
    bounds,
    time_range : int,
    crs: int,
    tile: str,
    number: str,
    verbose: bool,
):
    """
    Load and save the ESA World Cover data to the working directory.

    Parameters
    ----------
    data_path : str
        The path to save the data to.
    items_in_timestamp : list[Item]
        The items to download.
    bands : list[str]
        The bands to download.
    buffered_bounds : box
        The buffered bounds to download the data for.
    bounds : box
        The bounds to download the data for.
    crs : int
        The CRS of the data.
    tile : str
        The tile of the data.
    number : str    
        The id number of sub-tile.
    verbose : bool
        Whether to print verbose output.
    """
    attributes = get_attributes('esa-worldcover')['data_attrs']
    _bands_10m = set(attributes['Band'].where(attributes['Spatial Resolution'] == 10).dropna().to_list())
    _dytpes = dict(zip(attributes["Band"], attributes["Data Type"]))

    box_10m_buffered = GeoBox.from_bbox(buffered_bounds, CRS.from_epsg(crs), resolution=10)

    sel_bands_s = set(bands)
    bands_10m = list(_bands_10m & sel_bands_s)

    out_path = os.path.join(data_path, f"{tile}_{number}_{time_range}_ESA_WC_v0.zarr.zip")

    if os.path.exists(out_path):
        if verbose:
            print(f"Skipping {time_range}, already exists", flush=True)
        return out_path

    if verbose:
        print(f"Loading {bands} for {time_range}", flush=True)

    start_time = time.time()
    parameters = {
        "items": items_in_timestamp,
        "patch_url": pc.sign,
        "bands": bands_10m,
        "dtype": _dytpes,
        "chunks": {"time": 1, "x": -1, "y": -1},
        "groupby": "solar_day",
        "resampling": "nearest",
        "fail_on_error": True,
        "geobox": box_10m_buffered,
    }
    for _ in range(5):
        try:
            esa_wc = stac.load(**parameters).compute()
            if verbose:
                print(  # TODO: to be replaced with logging
                    f"Time taken for {time_range} for {out_path}: {time.time() - start_time:.0f} s",
                    flush=True,
                )
            break
        except (WarpOperationError, RasterioIOError) as e:
            print(f"Error creating {bands} for {time_range}: {e}", flush=True)
            time.sleep(5)

    esa_wc = esa_wc.sel(x=slice(bounds[0], bounds[2]),
                        y=slice(bounds[3], bounds[1]))

    cube_attrs = {
        'EPSG': crs,
        'UTM Tile': tile,
        'Bucket': tile + '_' + number,
        'Creation Time': datetime.datetime.now().isoformat(),
    }
    esa_wc.attrs = {**base_attrs(), **cube_attrs}

    esa_wc = esa_wc.squeeze().drop_vars(["time"])  # , "spatial_ref"])
    esa_wc = esa_wc.rename_vars({"map": f"esa_worldcover_{time_range}"})
    esa_wc = esa_wc.chunk({"x": -1, "y": -1})
    esa_wc = esa_wc.astype("uint8")
    esa_wc[f"esa_worldcover_{time_range}"].attrs = get_attributes('esa-worldcover')['esa_worldcover']

    store = zarr.ZipStore(out_path, mode="x", compression=zipfile.ZIP_BZIP2)
    esa_wc.to_zarr(store, mode="w-", consolidated=True)
    store.close()

    return out_path

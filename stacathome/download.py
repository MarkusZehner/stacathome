import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve

import planetary_computer as pc
import zarr
from odc.stac import load
from rasterio.errors import RasterioIOError, WarpOperationError
from shapely import box, transform

from .asset_specs import get_attributes, get_resampling_per_band, supported_mspc_collections
from .utils import filter_items_to_data_coverage, get_transform, run_with_multiprocessing


def download_request_from_probe(probe_request, sel_bands, workdir, fname=None):
    file_list = []
    for collection_name in supported_mspc_collections().keys():
        if collection_name in probe_request and collection_name in sel_bands:
            loc_name, item_collections, request_box = probe_request[collection_name]
            # loc_name = fname if fname else loc_name
            out_path = os.path.join(workdir, loc_name + collection_name + '_')
            bands = sel_bands[collection_name]
            files_ = get_cube_part(
                item_collections,
                collection=collection_name,
                request_geobox=request_box,
                selected_bands=bands,
                out_dir=out_path,
            )
            file_list.extend(files_)
    return file_list


def get_cube_part(items, collection, request_geobox, selected_bands, out_dir, asset_bin_size=50):
    selected_bands = set(selected_bands) & set(get_attributes(collection)['data_attrs']['Band'])

    crs = int(str(request_geobox.crs).split(':')[-1])
    transformed_box = transform(box(*request_geobox.boundingbox), get_transform(crs, 4326))
    items = filter_items_to_data_coverage(items, transformed_box, sensor=collection)

    s2_attributes = get_attributes(collection)['data_attrs']
    _s2_dytpes = dict(zip(s2_attributes["Band"], s2_attributes["Data Type"]))
    if collection == 'sentinel-2-l2a':
        resampling = get_resampling_per_band(
            abs(int(request_geobox.resolution.x)), bands=selected_bands, collection=collection
        )
    else:
        resampling = "nearest"

    # subset items if asset_bin_size is not None
    file_list = []
    if asset_bin_size is not None and len(items) > asset_bin_size:
        items = [items[i : i + asset_bin_size] for i in range(0, len(items), asset_bin_size)]
        for asset_bin_nr, item_list in enumerate(items):
            out_dir_ = out_dir + f"asset_bin_{str(asset_bin_nr * asset_bin_size).zfill(4)}.zarr.zip"
            file_list.append(out_dir_)
            if os.path.exists(out_dir_):
                print(f"Asset bin {asset_bin_nr} already exists, skipping")
                continue
            run_with_multiprocessing(
                save_stac_to_zarr_zip,
                items=item_list,
                bands=selected_bands,
                dtype=_s2_dytpes,
                out_path=out_dir_,
                geobox=request_geobox,
                resampling=resampling,
            )

    else:
        out_dir_ = out_dir + "all_assets.zarr.zip"
        file_list.append(out_dir_)
        if os.path.exists(out_dir_):
            print("Asset already exists, skipping")
            return file_list
        run_with_multiprocessing(
            save_stac_to_zarr_zip,
            items=items,
            bands=selected_bands,
            dtype=_s2_dytpes,
            out_path=out_dir_,
            geobox=request_geobox,
            resampling=resampling,
        )

    return file_list


def save_stac_to_zarr_zip(items, out_path, bands, dtype, geobox=None, boundingbox=None, resampling="nearest"):
    parameters = {
        "items": items,
        "patch_url": pc.sign,
        "bands": bands,
        "dtype": dtype,
        # 'chunks': {"time": -1, "x": -1, "y": -1},
        "groupby": "solar_day",
        "resampling": resampling,
        "fail_on_error": True,
    }
    if geobox:
        parameters["geobox"] = geobox
    if boundingbox:
        parameters["bbox"] = boundingbox

    for i in range(5):
        try:
            data = load(**parameters).compute()
            break
        except (WarpOperationError, RasterioIOError):
            print(f"Error creating Zarr: retry {i}", flush=True)
            time.sleep(5)

    store = zarr.ZipStore(out_path, mode="x")
    data.to_zarr(store, mode="w-")
    store.close()


def get_asset(href, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        return
    try:
        urlretrieve(pc.sign(href), save_path)
    except (KeyboardInterrupt, SystemExit):
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception as e:
                print(f"Error during cleanup of file {save_path}:", e)
    except Exception as e:
        print(f"Error downloading {href}:", e)


def download_assets_parallel(asset_list, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda args: get_asset(*args), asset_list)

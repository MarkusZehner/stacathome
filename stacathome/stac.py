from os import listdir, path as os_path
import sys
from copy import deepcopy

import geopandas as gpd
from rasterio import open as rio_open
from rasterio.errors import RasterioIOError, WarpOperationError
from rasterio.windows import Window
from pystac import ItemCollection
from pystac_client import Client as pystacClient
from shapely.geometry import shape as s_shape



def get_unique_elements(lists_of_objects):
    ids = [i.id for i in lists_of_objects[0]]
    out = list(lists_of_objects[0])
    for items in lists_of_objects[1:]:
        for i in items:
            if i.id not in ids:
                out.append(i)
                ids.append(i.id)
    return out



def items_to_dataframe(items, to_crs=None):
    """
    Convert a list of STAC items to a GeoDataFrame

    Parameters
    ----------
    items : list
        List of STAC items
    crs : str, default 'EPSG:4326' as the stac items are in WGS84
        CRS of the GeoDataFrame

    Returns
    -------
    gdf : GeoDataFrame
        GeoDataFrame with the STAC items
    """
    d = []
    asset_id = []
    geometry = []
    assets = []
    for i in items:
        d.append(i.properties)
        asset_id.append(i.id)
        assets.append(list(i.assets.keys()))
        geometry.append(s_shape(i.geometry))
    gdf = gpd.GeoDataFrame(d, geometry=geometry, crs="EPSG:4326")
    gdf["asset_id"] = asset_id
    gdf["assets"] = assets
    gdf["asset_items"] = items
    if to_crs is not None:
        gdf = gdf.to_crs(f"EPSG:{to_crs}")
    return gdf

def get_size_of_list_elements(lst):
    sizes = [(i, sys.getsizeof(item)) for i, item in enumerate(lst)]
    total_size = sum(size for _, size in sizes)

    for idx, size in sizes:
        print(f"Item {idx} size: {size} bytes")

    print(f"Total size of the list elements: {total_size} bytes")
    return total_size


def check_assets(items):
    try:
        not_found = []
        read_failed = []
        for i in items:
            for a in i.assets:
                path = i.assets[a].href
                if not path.startswith("/Net") or not os_path.exists(path):
                    not_found.append(path)
                else:
                    try:
                        src = rio_open(path)
                        src.read(
                            1,
                            window=Window(
                                src.width - 256, src.height - 256, src.width, src.height
                            ),
                        )
                    except (RasterioIOError, WarpOperationError, Exception) as e:
                        read_failed.append(path)
        # return None
        return not_found, read_failed
    except Exception as e:
        return e
    

def check_parallel_request(items, requested_bands, path):
    """
    Check the items from a parallel request against the local assets.

    Parameters
    ----------
    items: list
        The items to check.

    Returns
    -------
    filtered_requests: list
        The items that are not locally available.
    """
    assets = []
    for p in items:
        for i in p:
            for a in requested_bands:
                save_path = os_path.join(*[path] + i.assets[a].href.split("/")[-6:])
                if not os_path.exists(save_path):
                    assets.append((i.assets[a].href, save_path))
    return list(set(assets))


def get_all_local_assets(out_path, collection="sentinel-2-l2a", requested_bands=None):
    files = listdir(out_path)
    # S--2 specific naming scheme
    files = [f[:27] + f[33:-5] for f in files if f.endswith(".SAFE")]

    stac = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystacClient.open(stac)
    local_items = []

    files_stack = [files[i : i + 100] for i in range(0, len(files), 100)]
    for s in files_stack:
        local_items.extend(
            catalog.search(
                ids=s,
                collections=[collection],
            ).item_collection()
        )
    all_local_avail_items, _ = check_request_against_local(
        local_items, out_path, requested_bands=requested_bands, report=False
    )
    return all_local_avail_items


def check_request_against_local(items, out_path, requested_bands=None, report=False):
    """
    Check which of the requested assets are already downloaded and which are missing.
    list of available assets can be directly passed to odc-stac.
    """
    measurement_assets = [
        "AOT",
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B11",
        "B12",
        "B8A",
        "SCL",
        "WVP",
    ]
    ingnore_assets = [
        "visual",
        "preview",
        "safe-manifest",
        "granule-metadata",
        "inspire-metadata",
        "product-metadata",
        "datastrip-metadata",
        "tilejson",
        "rendered_preview",
    ]

    if not requested_bands:
        requested_bands = measurement_assets
    else:
        assert set(requested_bands) <= set(measurement_assets)

        ingnore_assets = ingnore_assets + [
            a for a in measurement_assets if a not in requested_bands
        ]

    if type(items) == ItemCollection:
        local_items = items.clone()
        local_items = list(local_items)
    elif type(items) == list:
        local_items = deepcopy(items)
    idx_to_pop = []
    not_downloaded_items = []
    to_download = []
    missing_assets = 0
    for i in range(len(local_items)):
        downloaded = True
        if len(local_items[i].assets.keys()) > 0:
            if os_path.exists(
                os_path.join(
                    out_path,
                    next(iter(local_items[i].assets.values()))
                    .href.split("?")[0]
                    .split("/")[-6],
                )
            ):
                for b in requested_bands:
                    try:
                        # check if the file is already downloaded
                        # if yes, add path to local_items
                        save_path = os_path.join(
                            *[out_path]
                            + local_items[i]
                            .assets[b]
                            .href.split("?")[0]
                            .split("/")[-6:]
                        )
                        # check_sentinel2_data_exists_with_min_size(save_path):
                        if os_path.exists(save_path):
                            local_items[i].assets[b].href = save_path
                        else:
                            to_download.append(
                                (local_items[i].assets[b].href, save_path)
                            )
                            missing_assets += 1
                            downloaded = False
                            del local_items[i].assets[b]
                    except KeyError as e:
                        pass
                        # print(f'Asset {b} not found in item {local_items[i].id}')
            else:
                for b in requested_bands:
                    save_path = os_path.join(
                        *[out_path]
                        + local_items[i].assets[b].href.split("?")[0].split("/")[-6:]
                    )
                    to_download.append((local_items[i].assets[b].href, save_path))
                    del local_items[i].assets[b]
                    missing_assets += 1
                downloaded = False

        for b in ingnore_assets:
            try:
                del local_items[i].assets[b]
            except KeyError as e:
                pass

        if not downloaded:
            not_downloaded_items.append(local_items[i])

        # if set is only ignore_assets, then we don't have the data
        if (
            set(local_items[i].assets.keys()) == set(ingnore_assets)
            or len(local_items[i].assets.keys()) == 0
        ):
            idx_to_pop.append(i)

    for i in idx_to_pop[::-1]:
        local_items.pop(i)

    if report:
        if len(not_downloaded_items) == 0:  # should be a logger
            print("All data already downloaded.")
        else:
            print(
                f"{missing_assets} missing assets of {
                len(not_downloaded_items)} items to download."
            )

    return local_items, to_download
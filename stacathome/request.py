import json
import os
import time

from collections import defaultdict
import numpy as np
from odc.geo.geobox import GeoBox
from pystac import Item
from pystac_client import Client
from pystac_client.exceptions import APIError
from shapely import box, buffer, transform, union, union_all

from .asset_specs import (get_stac_filter_arg, supported_mspc_collections, get_asset_box_and_transform,
                          contains_in_native_crs, get_asset_crs, transform_asset_bbox,
                          centroid_distance_in_native_crs, intersection_area_percent_in_native_crs)
from .utils import cut_box_to_edge_around_coords, get_transform, get_utm_crs_from_lon_lat, time_range_parser


def probe_request(
    point_wgs84,
    distance_in_m: int = 10000,
    collection: str | list[str] = 'sentinel-2-l2a',
    url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    return_box=False,
    limit=12,
):
    """
    Function to probe a STAC Catalog for a given area, defined by a point and a edge length in meters.
    returns either one tile which contains completely or a list of tiles that intersect with the area.
    """
    return_dict = {}
    for collection_name in supported_mspc_collections().keys():
        if collection == collection_name or collection_name in collection:
            # if collection_name == 'esa-worldcover':
            #     return_dict[collection_name] = 'Does not require a probe, no temporal dimension.'
            #     continue
            return_dict[collection_name] = probe_collection(
                point_wgs84, distance_in_m, collection_name, url, return_box, limit
            )
            collection.pop(collection.index(collection_name))

    if collection:
        print(f"Collection {collection} not supported yet.")

    return return_dict


def probe_collection(
    point_wgs84,
    distance_in_m: int = 10000,
    collection: str = 'sentinel-2-l2a',
    url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    return_box=False,
    limit=12,
):
    crs_code = get_utm_crs_from_lon_lat(point_wgs84.x, point_wgs84.y)
    tr = get_transform(4326, crs_code)
    tr_back = get_transform(crs_code, 4326)
    point_utm = transform(point_wgs84, tr)
    distance_in_m = distance_in_m * 1.1
    bbox_wgs84 = transform(box(*buffer(point_utm, distance_in_m / 2).bounds), tr_back)

    query = request_stac_rest_api(
        collection=collection,
        url=url,
        intersects=bbox_wgs84,
        max_items=limit,
    )

    tile_keyword = get_stac_filter_arg(collection)
    if tile_keyword is None:
        if return_box:
            return query, bbox_wgs84
        return query

    # check if contained in any tile, if multiple choose the one with the smallest distance to the center
    tile_ids_contained = [
        (
            asset.properties[tile_keyword],
            get_asset_crs(asset, collection=collection),
            contains_in_native_crs(asset, bbox_wgs84, collection=collection),
            centroid_distance_in_native_crs(asset, bbox_wgs84, collection=collection),
        )
        for asset in query
    ]
    tile_ids, tile_crs, contained, distances = list(zip(*tile_ids_contained))

    # easy case: one tile fity request
    skip_funny_business = False
    if any(contained):
        tiles = list({x for x, y in zip(tile_ids, contained) if y})
        if len(tiles) > 1:
            print('Multiple tiles containing found, choosing the one with the smallest distance to the center')
            tiles = [tile_ids[distances.index(min(distances))]]
        skip_funny_business = True

    # if not, calculate intersection with all intersecting tiles
    # iteratively union them until the requested area is contained
    else:
        # request_area = bbox_wgs84.area
        tiles_area = {}
        tiles_bounds = {}
        most_found_crs = max(set(tile_crs), key=tile_crs.count)

        for asset in query:
            tile_id_iter = asset.properties[tile_keyword]
            if tile_id_iter not in tiles_area:
                tiles_area[tile_id_iter] = []
                tiles_bounds[tile_id_iter] = []

            tiles_area[tile_id_iter].extend(
                [intersection_area_percent_in_native_crs(asset, bbox_wgs84, collection=collection)]
            )
            asset_bbox, _ = get_asset_box_and_transform(asset, collection=collection)
            asset_crs = get_asset_crs(asset, collection)
            if asset_crs != most_found_crs:
                asset_bbox = transform(asset_bbox, get_transform(asset_crs, most_found_crs))
            tiles_bounds[tile_id_iter].extend([asset_bbox])

        for key in tiles_area.keys():
            tiles_area[key] = np.mean(tiles_area[key])

        tiles_area = {k: v for k, v in sorted(tiles_area.items(), key=lambda item: item[1], reverse=True)}

        # check if the zones are all within the same utm zone
        max_key = None
        # print(tile_crs)
        if len(set(tile_crs)) > 1:
            print(
                'Tiles are not in the same UTM zone returning tiles with percentage of area cover in the requested area as dict',
                tiles_area
            )
            # if not, we want to know which crs covers most
            grouped_sums = defaultdict(float)
            for key, value in tiles_area.items():
                grouped_sums[key[:2]] += value
            max_key = max(grouped_sums, key=grouped_sums.get)
            best_matching_crs = [code for code in set(tile_crs) if str(code)[-2:] == max_key]

        iterative_shape = None
        tiles = []
        for key in tiles_bounds.keys():
            if not iterative_shape:
                iterative_shape = max(set(tiles_bounds[key]), key=tiles_bounds[key].count)
            iterative_shape = union(iterative_shape, (max(set(tiles_bounds[key]), key=tiles_bounds[key].count)))
            tiles.append(key)
            if iterative_shape.contains(bbox_wgs84):
                break

    if return_box:
        tilelist = tiles if isinstance(tiles, list) else list(tiles.keys())
        # print(tilelist)
        boxes = {}
        for t in tilelist:
            box_t = [
                transform_asset_bbox(asset, collection=collection)
                for asset in query
                if asset.properties[tile_keyword] == t
            ]
            boxes[t] = max(set(box_t), key=box_t.count)

        if len(set(tile_crs)) > 1 and not skip_funny_business:
            # if now multiple crs are found, we need to enlarge one of the boxes in the correct crs,
            # so that the final grid is consistent with largest amount of data
            use_this_tile = [k for k in list(boxes.keys()) if str(k)[:2] == max_key]
            use_this_box = boxes[use_this_tile[0]]
            tr = get_transform(4326, best_matching_crs[0])
            tr_back = get_transform(best_matching_crs[0], 4326)
            box_utm = transform(use_this_box, tr)
            bounds = list(box_utm.bounds)
            bounds = [bounds[0] - distance_in_m, bounds[1] - distance_in_m,
                      bounds[2] + distance_in_m, bounds[3] + distance_in_m]
            bounds = [round(b) for b in bounds]
            bounds = box(*[bounds[0] - distance_in_m, bounds[1] - distance_in_m,
                           bounds[2] + distance_in_m, bounds[3] + distance_in_m])
            boxes = {use_this_tile[0]: transform(bounds, tr_back)}

        return tiles, boxes

    return tiles


def build_request_from_probe(
    center_point, time_range, edge_length_m, target_res_m, probe_dict, match_on_s2grid=True, save_dir=None, fname=None,
):
    requests = {}
    if match_on_s2grid and 'sentinel-2-l2a' not in probe_dict:
        raise ValueError("Match on S2 grid requires a sentinel-2-l2a probe")
    if 'sentinel-2-l2a' in probe_dict:
        requests['sentinel-2-l2a'] = build_request_from_probe_collection(
            center_point,
            time_range['sentinel-2-l2a'],
            edge_length_m,
            target_res_m,
            probe_dict['sentinel-2-l2a'],
            collection='sentinel-2-l2a',
            save_dir=save_dir,
            fname=fname,
        )
        if match_on_s2grid:
            request_box_match = requests['sentinel-2-l2a'][2]
    if 'modis-13Q1-061' in probe_dict:
        if not match_on_s2grid:
            request_box_match = None
        requests['modis-13Q1-061'] = build_request_from_probe_collection(
            center_point,
            time_range['modis-13Q1-061'],
            edge_length_m,
            target_res_m,
            probe_dict['modis-13Q1-061'],
            collection='modis-13Q1-061',
            request_box=request_box_match,
            save_dir=save_dir,
            fname=fname,
        )
    if 'esa-worldcover' in probe_dict:
        if not match_on_s2grid:
            request_box_match = None
        requests['esa-worldcover'] = build_request_from_probe_collection(
            center_point,
            time_range['esa-worldcover'],
            edge_length_m,
            target_res_m,
            probe_dict['esa-worldcover'],
            collection='esa-worldcover',
            request_box=request_box_match,
            save_dir=save_dir,
            fname=fname,
        )
    # TODO: add other sensors
    return requests


def build_request_from_probe_collection(
    center_point,
    time_range,
    edge_length_m,
    target_res_m,
    probe_values,
    collection,
    loc_name=None,
    request_box=None,
    save_dir=None,
    fname=None
):
    tile_ids = probe_values[0]
    # if not all([i.startswith(tile_ids[0][:2]) for i in tile_ids]):
    #     raise ValueError("All tiles must be from the same UTM zone")
    tiles = '_'.join(tile_ids)
    time_range = time_range_parser(time_range)
    time_range_str = str(time_range).replace('/', '_')
    if fname is None:
        fname = 'customcube'
    loc_name = (
        f"{fname}_{center_point.y:.2f}_{center_point.x:.2f}_{time_range_str}_{tiles}_"
        if loc_name is None
        else loc_name
    )

    if not request_box:
        if collection in ['modis-13Q1-061', 'esa-worldcover']:
            raise ValueError(
                f"Collection {collection} requires a matched request box, native resolution not yet implemented"
            )
        n_pix_edge = int(edge_length_m // target_res_m)
        crs_zone = int(probe_values[0][0][0:2]) + 32600 if center_point.y > 0 else 32700
        utm_boxes = [transform(i, get_transform(4326, crs_zone)) for i in probe_values[1].values()]
        if len(utm_boxes) > 1:
            utm_boxes = union_all(utm_boxes)
        else:
            utm_boxes = utm_boxes[0]
        full_tile_box = GeoBox.from_bbox(utm_boxes.bounds, crs=crs_zone, resolution=target_res_m)
        request_box = cut_box_to_edge_around_coords(
            transform(center_point, get_transform(4326, int(crs_zone))), full_tile_box, n_pix_edge
        )

    if request_box is not None and collection == 'modis-13Q1-061':
        request_box = GeoBox.from_bbox(
            box(*request_box.boundingbox).buffer((250 // target_res_m + 1) * target_res_m).bounds,
            crs=request_box.crs,
            resolution=target_res_m,
        )

    item_collections = []
    for tile_id in tile_ids:
        item_collections.extend(request_data_by_tile(tile_id, time_range, collection=collection, save_dir=save_dir))

    return loc_name, item_collections, request_box


def request_data_by_tile(
    tile_id: str,
    time_range: str | int,
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
    time_range : str | int
        The time range time of the query. int for year, str for date range as format "YYYY-MM-DD/YYYY-MM-DD".
        See time_range_parser.
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
    if isinstance(time_range, str):
        time_range_str = time_range.replace('/', '_')
    else:
        time_range_str = str(time_range)
    out_path = os.path.join(save_dir, f"{collection}_{tile_id}_" f"{time_range_str}_query.json") if save_dir else None
    if out_path and os.path.exists(out_path):
        with open(out_path) as json_file:
            query_dict = json.load(json_file)
    else:
        filter_arg = get_stac_filter_arg(collection)
        query_dict = request_stac_rest_api(
            collection=collection, url=url, datetime=time_range, query={filter_arg: dict(eq=tile_id)}
        ).to_dict()

        if query_dict == {}:
            print(f"Failed to get data for {tile_id} {time_range}.", flush=True)
            return []

        if out_path:
            with open(out_path, "w") as json_file:
                json.dump(query_dict, json_file, indent=4)

    return [Item.from_dict(feature) for feature in query_dict["features"]]


def request_data_by_bbox(bbox,
                         time_range,
                         save_dir,
                         url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
                         collection: str = "sentinel-2-l2a",
                         verbose=False) -> list[Item]:
    """
    Request data from the Planetary Computer API by bounding box and time range.

    Parameters
    ----------
    bbox : BoundingBox
        The bounding box to filter the items by.
    time : int
        The year to filter the items by.
    save_dir : str
        The directory to save the downloaded data to.
    url : str, optional
        The API url to query, by default "https://planetarycomputer.microsoft.com/api/stac/v1".
    collection : str, optional
        The collection to query, by default "sentinel-2-l2a".
    verbose : bool, optional
        Whether to print verbose output, by default False.

    Returns
    -------
    list[Item]
        List of pystac.Item objects.
    """
    if isinstance(time_range, str):
        time_range_str = time_range.replace('/', '_')
    else:
        time_range_str = str(time_range)
    out_path = os.path.join(save_dir, f"{collection}_{bbox.bounds}_" f"{time_range_str}_query.json") if save_dir else None
    if out_path and os.path.exists(out_path):
        with open(out_path) as json_file:
            query_dict = json.load(json_file)
    else:
        query_dict = request_stac_rest_api(
            collection=collection,
            url=url,
            datetime=time_range,
            intersects=bbox,
        ).to_dict()

        if query_dict == {}:
            print(f"Failed to get data for {bbox.bounds} {time_range}.", flush=True)
            return []

        if out_path:
            with open(out_path, "w") as json_file:
                json.dump(query_dict, json_file, indent=4)

    return [Item.from_dict(feature) for feature in query_dict["features"]]


def request_stac_rest_api(
    collection: str = 'sentinel-2-l2a',
    url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    max_retry: int = 5,
    **kwargs,
):
    if isinstance(collection, str):
        collection = [collection]
    for _ in range(max_retry):
        try:
            query = Client.open(url).search(collections=collection, **kwargs).item_collection()
            break
        except APIError:
            time.sleep(3)
            query = None
    if query is None:
        raise ValueError("Failed to get data from the API")
    return query

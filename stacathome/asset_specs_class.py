import time
import math
from pathlib import Path
# from pyproj import CRS
from shapely import box, transform  # Polygon
from functools import partial
# import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any  # List
from pystac import Item
from pystac_client import Client
from pystac_client.exceptions import APIError
import planetary_computer as pc
import logging

from .utils import get_transform

import os
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve
from odc.stac import load
from rasterio.errors import RasterioIOError, WarpOperationError
# import geopandas as gpd
from collections import defaultdict, Counter
from shapely import unary_union
import numpy as np

from odc.geo.geobox import GeoBox
import re


logging.basicConfig(
    level=logging.INFO,  # Set to WARNING or ERROR in production
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Helper to get the most common element
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


def get_asset(href: Path, save_path: Path, signer: callable = pc.sign):
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


def parse_place(request_place: str | list | tuple | dict):
    """
    the place for a stac request in different formats:
    - str: request the bounding box as "minx,miny,maxx,maxy"
    - str: "lon, lat" in EPSG:4326 either as decimals or degree

    """
    raise NotImplementedError


class STACProvider():
    def __init__(self, url: str = "https://planetarycomputer.microsoft.com/api/stac/v1", sign: callable = pc.sign, **kwargs):
        self.url = url
        self.sign = sign
        self.client = Client.open(self.url)
        self.extra_attributes = kwargs
        self.query = None

        # possible to add checks if cql2 filter, which collections are present...

        # should the classes have a default pipe-through if all their functions if all kwargs are matching?

    def request_items(self, collection: list[str], request_time: str, request_place: any,
                      limit: int = None, max_retry: int = 5, **kwargs):
        if isinstance(collection, str):
            collection = [collection]
        query = None
        for _ in range(max_retry):
            try:
                query = self.client.search(collections=collection,
                                           datetime=request_time,
                                           intersects=request_place,
                                           max_items=limit,
                                           ** kwargs).item_collection()
                break
            except APIError as e:
                logging.warning(f"APIError: Retrying because of {e}")
                if "429" in str(e):
                    # Too many requests, wait and retry
                    time.sleep(3)
        if query is None:
            raise ValueError("Failed to get data from the API")
        return query

    def download_granules_to_file(self, href_path_tuples: list[tuple]):
        download_assets_parallel(href_path_tuples, signer=self.sign)

    def download_cube(self, parameters):
        data = None
        for i in range(5):
            try:
                data = load(**parameters).load()
                break
            except (WarpOperationError, RasterioIOError):
                print(f"Error creating cube: retry {i}", flush=True)
                time.sleep(5)
        if data is None:
            raise ValueError("Failed to create cube")
        return data


class STACRequest():
    def __init__(self, collections: list[str], request_place: any, request_time: any):
        AVAIL_COLLECTIONS = set(ItemProcessor._config.keys())
        # add other provider collection here
        self.stac_providers = defaultdict(list)
        unmatched_collections = set()

        if isinstance(collections, str):
            collections = [collections]
        for c in collections:
            if c in AVAIL_COLLECTIONS:
                self.stac_providers[STACProvider()].append(c)
            # add other providers here
            else:
                unmatched_collections.add(c)
        if unmatched_collections:
            logging.warning(f"Some collections were not matched: {unmatched_collections}")

        self.request_place = request_place
        self.request_time = parse_time(request_time)

        # will get a list of sensors, lat long or shape or bbox and time
        # finds the needed tiles for a given request by probing the STAC
        # should nudge the provided location towards grid matching the data sources if not specified otherwise
        # gathers all items and returns them with geoboxes per gridding in the sensors

    def collect_covering_tiles(self, item_limit=12):
        returned_items_per_collection = defaultdict(list)
        for provider, collections in self.stac_providers.items():
            for collection in collections:
                items = provider.request_items(collection=collection,
                                               request_time=self.request_time,
                                               request_place=self.request_place,
                                               limit=item_limit,
                                               max_retry=5)
                if len(items) < item_limit:
                    logging.warning(f"Not enough items found for {collection} in {self.request_place} "
                                    f"and {self.request_time}")

                by_tile = defaultdict(list)
                proc = ItemProcessor()

                for i in items:
                    item = proc.get_item_processor(i)
                    by_tile[item.get_tile_id()].append([
                        item.get_crs(),
                        item.contains_shape(self.request_place),
                        item.centroid_distance_to(self.request_place),
                        item.overlap_percentage(self.request_place),
                        item.get_bbox(),
                    ])

                # Reduce each group using majority voting
                by_tile_filtered = [
                    [tile_id] + [most_common(attr) for attr in zip(*vals)]
                    for tile_id, vals in by_tile.items()
                ]

                # First, try finding a containing item
                best = resolve_best_containing(by_tile_filtered)
                if best:
                    found_tiles = [best]
                else:
                    found_tiles = merge_to_cover(by_tile_filtered, self.request_place)

                tile_ids = [t[0] for t in found_tiles]
                crs = found_tiles[0][1]
                aligned_box = item.snap_bbox_to_grid(transform(self.request_place,
                                                               get_transform(4326, crs)).bounds)

                geobox_dict = proc.get_item_processor(items[0]).get_geobox(
                    bbox=aligned_box,
                    crs=crs
                )

                returned_items_per_collection[collection].extend([tile_ids, geobox_dict])
        return dict(returned_items_per_collection)

    def align_input_location_with_grid(self):
        raise NotImplementedError

    def request_items(self, ):
        returned_items_per_collection = defaultdict(list)
        for provider, collections in self.stac_providers.items():
            for collection in collections:
                items = provider.request_items(collection=collection,
                                               request_time=self.request_time,
                                               request_place=self.request_place,
                                               limit=5,
                                               max_retry=5)
                returned_items_per_collection[collection].extend(items)
        return returned_items_per_collection


@dataclass
class Band:
    name: str
    data_type: str
    nodata_value: int | float
    spatial_resolution: int | float
    continuous_measurement: bool
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str, data_type: str, nodata_value: int, spatial_resolution: int,
                 continuous_measurement: bool, **kwargs):
        self.name = name
        self.data_type = data_type
        self.nodata_value = nodata_value
        self.spatial_resolution = spatial_resolution
        self.continuous_measurement = continuous_measurement
        self.extra_attributes = kwargs  # Store additional attributes here

    def to_dict(self):
        """Converts the Band object into a dictionary for DataFrame conversion."""
        base_dict = {
            "Name": self.name,
            "Data Type": self.data_type,
            "NoData Value": self.nodata_value,
            "Spatial Resolution": self.spatial_resolution,
            "Continuous Measurement": self.continuous_measurement,
        }
        # Merge extra attributes into the dictionary
        return {**base_dict, **self.extra_attributes}


class STACItem:
    def __init__(self, item):
        self.item = item
        self.collection = item.collection_id

    def get_crs(self):
        """To be implemented in subclasses"""
        raise NotImplementedError

    def get_tile_id(self):
        """To be implemented in subclasses"""
        raise NotImplementedError

    def get_bbox(self):
        """To be implemented in subclasses"""
        raise NotImplementedError

    def get_geobox(self):
        """To be implemented in subclasses"""
        raise NotImplementedError

    def get_assets_as_bands(self):
        """To be implemented in subclasses"""
        raise NotImplementedError

    def get_transform(self, from_crs=None, to_crs=None):
        assert to_crs or from_crs, "define one of to_crs or from_crs"
        return get_transform(from_crs=from_crs or self.get_crs(), to_crs=to_crs or self.get_crs())

    def get_resampling_per_band(self, target_resolution):
        return {b.name: ("nearest"
                         if target_resolution <= b.spatial_resolution
                         else "bilinear"
                         if b.continuous_measurement
                         else "mode") for b in self.get_assets_as_bands()}

    def get_dtype_per_band(self, target_resolution):
        resampling = self.get_resampling_per_band(target_resolution)
        return {b.name: (b.data_type
                         if resampling[b.name] == 'nearest'
                         else "float32")
                for b in self.get_assets_as_bands()}

    def contains_shape(self, shape, shape_crs=4326):
        """
        Check if the item contains a given shape.
        Args:
            shape (Polygon): The shape to check against the item's bounding box.
        Returns:
            bool: True if the item contains the shape, False otherwise.
        """
        bbox = self.get_bbox()
        transformed_shape = transform(shape, get_transform(shape_crs, self.get_crs()))
        return bbox.contains(transformed_shape)

    def centroid_distance_to(self, shape, shape_crs=4326):
        """
        Calculate the distance from the item's centroid to a given shape.
        Args:
            shape (Polygon): The shape to calculate the distance to.
        Returns:
            float: The distance from the item's centroid to the shape.
        """
        item_centroid = self.get_bbox().centroid
        transformed_shape = transform(shape, get_transform(shape_crs, self.get_crs()))
        return item_centroid.distance(transformed_shape.centroid)

    def overlap_percentage(self, shape, shape_crs=4326):
        """
        Calculate the percentage of overlap between the item and a given shape.
        Args:
            shape (Polygon): The shape to calculate the overlap with.
        Returns:
            float: The percentage of overlap between the item and the shape.
        """
        item_bbox = self.get_bbox()
        transformed_shape = transform(shape, get_transform(shape_crs, self.get_crs()))
        intersection = item_bbox.intersection(transformed_shape)
        return intersection.area / transformed_shape.area


class Sentinel2L2AProcessor(STACItem):
    def __init__(self, item):
        super().__init__(item)

    def get_crs(self):
        return self.item.properties['proj:epsg']

    def get_tile_id(self):
        return self.item.properties['s2:mgrs_tile']

    def get_data_asset_keys(self, role="data"):
        key = list(self.item.get_assets(role=role).keys())
        assert len(key) > 0, "No data assets found!"
        return key

    def get_assets_as_bands(self):
        supported_bands_S2 = ["B01",
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
                              "SCL"]
        present_bands = self.get_data_asset_keys()
        present_supported_bands = sorted(set(supported_bands_S2) & set(present_bands))

        bands = []
        for b in present_supported_bands:
            if b == 'SCL':
                bands.append(
                    Band(b, "uint8", 0, self.item.assets[b].extra_fields['gsd'], False,
                         long_name="Scene Classification Layer",
                         flag_meanings=[
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
                        flag_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    )
                )
            else:
                bands.append(Band(b, "uint16", 0, self.item.assets[b].extra_fields['gsd'], True,
                                  long_name=self.item.assets[b].extra_fields['eo:bands'][0]['description']
                                  ))

        return bands

    def get_bbox(self):
        key = self.get_data_asset_keys()[0]
        return box(*self.item.assets[key].extra_fields['proj:bbox'])

    def snap_bbox_to_grid(self, bbox, grid_size=60):
        xmin, ymin, xmax, ymax = bbox
        xmin_snapped = math.floor(xmin / grid_size) * grid_size
        ymin_snapped = math.floor(ymin / grid_size) * grid_size
        xmax_snapped = math.ceil(xmax / grid_size) * grid_size
        ymax_snapped = math.ceil(ymax / grid_size) * grid_size
        return box(xmin_snapped, ymin_snapped, xmax_snapped, ymax_snapped)

    def get_geobox(self, bbox, crs):
        bands = self.get_assets_as_bands()
        res = defaultdict(list)
        for b in bands:
            res[GeoBox.from_bbox(bbox.bounds, crs=crs, resolution=int(b.spatial_resolution))].append(b.name)

        return dict(res)

        # this shoudl return a dict of geoboxes with the list of which bands belong to which geobox
        # pass

    # class Modis13Q1Processor(AssetProcessor):
    #     def get_crs(self):
    #         return CRS.from_wkt(self.asset.properties['proj:wkt2'])

    #     def get_box_and_transform(self, from_crs=4326):
    #         to_crs = self.get_crs()
    #         bbox = Polygon(self.asset.properties['proj:geometry']['coordinates'][0])
    #         tr = self.get_transform(from_crs, to_crs)
    #         return bbox, tr

    # class ESAWorldcoverProcessor(AssetProcessor):
    #     def get_crs(self):
    #         return self.asset.properties['proj:epsg']

    #     def get_box_and_transform(self, from_crs=4326):
    #         to_crs = self.get_crs()
    #         bbox = box(*self.asset.bbox)
    #         tr = self.get_transform(from_crs, to_crs)
    #         return bbox, tr


class ItemProcessor:
    _config = {
        "sentinel-2-l2a": {
            "processor": Sentinel2L2AProcessor,
            "tilename": "s2:mgrs_tile",
        },
        # "modis-13Q1-061": {
        #     "processor": Modis13Q1Processor,
        #     "tilename": "modis:tile-id",
        # },
        # "esa-worldcover": {
        #     "processor": ESAWorldcoverProcessor,
        #     "tilename": "esa_worldcover:product_tile",
        # },
    }

    @staticmethod
    def get_item_processor(item):
        processor_cls = ItemProcessor._config.get(item.collection_id, {}).get("processor", STACItem)
        return processor_cls(item)

    @staticmethod
    def get_item_tilename(collection_name):
        if isinstance(collection_name, Item):
            collection_name = collection_name.collection_id
        return ItemProcessor._config.get(collection_name, {}).get("tilename", None)

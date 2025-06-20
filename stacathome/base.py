from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Any
import logging
import math
from datetime import datetime, timedelta
from pystac import Item
from odc.geo.geobox import GeoBox
import numpy as np
import xarray as xr
from shapely import transform, box
from asf_search import Products
from earthaccess.results import DataGranule

from stacathome.generic_utils import get_transform
from stacathome.providers import STACProvider, ASFProvider, EarthAccessProvider


logging.basicConfig(
    level=logging.INFO,  # Set to WARNING or ERROR in production
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# class BaseMetadataProcessor(ABC):
#     """
#     Basic processor implementing general methods, to be adjusted for specific collections
#     """
#     provider = None
#     collection: str = None

#     cubing: str = 'custom'

#     supported_bands: list = None
#     all_bands: dict = {}
#     tilename: str = None
#     datetime_id: str = None
#     overlap: bool = False
#     gridded: bool = False
#     x: str = 'x'
#     y: str = 'y'

#     # Init
#     @abstractmethod
#     def __init__(self, item):
#         pass

#     # classmethods
#     @classmethod
#     def get_supported_bands(cls) -> list:
#         """Return image data file names from collection as list"""
#         return cls.supported_bands

#     @classmethod
#     def get_tilename_key(cls) -> str:
#         """Return image data tile names from collection as list"""
#         return cls.tilename

#     @abstractmethod
#     @classmethod
#     def request_items(cls) -> list:
#         """Request metadata of available items from collection of a provider as list, will return provider specific result objects"""
#         pass

#     @abstractmethod
#     @classmethod
#     def download_tiles_to_file(cls):
#         pass

#     # specific methods
#     @abstractmethod
#     def snap_bbox_to_grid(self, bbox):
#         raise NotImplementedError

#     @abstractmethod
#     def does_cover_data(self):
#         raise NotImplementedError

#     @abstractmethod
#     def get_data_coverage_geometry(self):
#         raise NotImplementedError

#     @abstractmethod
#     def get_crs(self):
#         raise NotImplementedError

#     @abstractmethod
#     def get_bbox(self):
#         raise NotImplementedError

#     @abstractmethod
#     def get_geobox(self, bbox):
#         raise NotImplementedError

#     @abstractmethod
#     def load_cube(self, items, bands, geobox, split_by=100, chunks: dict = None):
#         raise NotImplementedError

#     # general methods
#     def get_assets_as_bands(self):
#         return self.all_bands.values()

#     def get_resampling_per_band(self, target_resolution):
#         return {b.name: ("nearest"
#                          if target_resolution <= b.spatial_resolution or math.isclose(target_resolution, b.spatial_resolution)
#                          else "bilinear"
#                          if b.continuous_measurement
#                          else "mode") for b in self.get_assets_as_bands()}

#     def get_dtype_per_band(self, target_resolution):
#         resampling = self.get_resampling_per_band(target_resolution)
#         return {b.name: (b.data_type
#                          if resampling[b.name] == 'nearest'
#                          else "float32")
#                 for b in self.get_assets_as_bands()}

#     def get_band_attributes(self, bands: list[str] | set[str]):
#         return {b: self.all_bands[b].to_dict() for b in bands if b in self.all_bands}

#     def solarday_offset_seconds(self, item):
#         item_centroid = self.__class__(item).get_bbox().centroid
#         item_centroid = transform(item_centroid, get_transform(self.get_crs(), 4326))
#         longitude = item_centroid.x
#         return int(longitude / 15) * 3600

#     def sort_items_by_datetime(self, items):
#         return sorted(items, key=lambda x: x.properties[self.datetime_id])

#     def split_items_keep_solar_days_together(self, items, split_by):
#         """
#         Split items by solar day.
#         Args:
#             items (list): List of items to split.
#         Returns:
#             dict: Dictionary with solar days as keys and lists of items as values.
#         """
#         rounded_datetimes = []
#         for item in items:
#             # modify the datetime here to represent solar daytime (offset by time * longitude)
#             dt = datetime.fromisoformat(item.properties[self.datetime_id].replace('Z', ''))
#             dt += timedelta(seconds=self.solarday_offset_seconds(item))
#             rounded_datetimes.append(dt.replace(hour=0, minute=0, second=0, microsecond=0))

#         sorted_datetimes = sorted(rounded_datetimes)
#         ranks = [sorted_datetimes.index(dt) + 1 for dt in rounded_datetimes]

#         # Group elements by rank
#         rank_groups = defaultdict(list)
#         for index, rank in enumerate(ranks):
#             rank_groups[rank].append(index)

#         # Convert the rank groups to a list of lists
#         ranked_elements = list(rank_groups.values())

#         batches = self.split_into_batches(ranked_elements, split_by)

#         # Output the batches
#         split_items = []
#         for batch in batches:
#             split_items.append([items[i] for i in batch])
#         return split_items

#     @staticmethod
#     def split_into_batches(elements, batch_size):
#         batches = []
#         current_batch = []
#         current_size = 0

#         for group in elements:
#             if current_size + len(group) > batch_size:
#                 batches.append(current_batch)
#                 current_batch = []
#                 current_size = 0
#             current_batch.extend(group)
#             current_size += len(group)

#         if current_batch:
#             batches.append(current_batch)

#         return batches

#     def centroid_distance_to(self, shape, shape_crs=4326):
#         """
#         Calculate the distance from the item's centroid to a given shape within the crs of the item.
#         Args:
#             shape (Polygon): The shape to calculate the distance to.
#         Returns:
#             float: The distance from the item's centroid to the shape.
#         """
#         item_centroid = self.get_bbox().centroid
#         transformed_shape = transform(shape, get_transform(shape_crs, self.get_crs()))
#         return item_centroid.distance(transformed_shape.centroid)

#     # unique per provider/collection
#     @staticmethod
#     def filter_product_iteration_links(data_granules: list):
#         """ECOSTRESS Earthaccess only"""
#         filtered = {}
#         for granule in data_granules:
#             native_id = granule['meta']['native-id']
#             base_name = native_id[:-3]
#             product_iteration = int(native_id[-2])
#             if base_name not in filtered or product_iteration > filtered[base_name][0]:
#                 filtered[base_name] = (product_iteration, granule)
#         return [v[1] for v in filtered.values()]


class EarthAccessProcessor(ABC):
    collection: str = ''
    supported_bands: list = []
    tilename: str = None
    provider = EarthAccessProvider()
    cubing = 'preferred'
    overlap = False
    gridded = False
    x = 'x'
    y = 'y'
    datetime_id = ''
    all_bands = {}

    def __init__(self, item: Item | DataGranule):
        self.item = item
        if isinstance(self.item, Item):
            self.collection = item.collection_id
        elif isinstance(self.item, DataGranule):
            c_ref = item['umm']['CollectionReference']
            self.collection = c_ref['ShortName'] + '.' + c_ref['Version']

    @classmethod
    def request_items(cls, _, request_time, request_place, max_items=-1, **kwargs):

        if 'maxResults' in kwargs:
            if max_items == -1:
                max_items = kwargs['maxResults']
            del kwargs['maxResults']

        if '.' in cls.collection:
            short_name, version = cls.collection.split('.')
        else:
            short_name = cls.collection
            version = None

        kwarg_dict = {
            'short_name': short_name,
            'version': version,
            'count' : max_items,

        }
        kwargs = kwargs | kwarg_dict

        return cls.provider.request_items(
            request_time,
            request_place,
            **kwargs,
        )

    @classmethod
    @abstractmethod
    def download_tiles_to_file(cls, path, items, bands, processes=4):
        raise NotImplementedError

    @staticmethod
    def filter_product_iteration_links(data_granules: list):
        filtered = {}
        for granule in data_granules:
            native_id = granule['meta']['native-id']
            base_name = native_id[:-3]
            product_iteration = int(native_id[-2])
            if base_name not in filtered or product_iteration > filtered[base_name][0]:
                filtered[base_name] = (product_iteration, granule)
        return [v[1] for v in filtered.values()]

    @classmethod
    def get_supported_bands(cls):
        return cls.supported_bands

    @classmethod
    def get_tilename_key(cls):
        return cls.tilename

    @abstractmethod
    def snap_bbox_to_grid(self, bbox):
        raise NotImplementedError

    def get_assets_as_bands(self):
        return self.all_bands.values()

    def does_cover_data(self, request_box, input_crs=None):
        if input_crs is not None:
            request_box = transform(request_box, get_transform(input_crs, self.get_crs()))
        return self.get_data_coverage_geometry().contains_properly(request_box)

    @abstractmethod
    def get_data_coverage_geometry(self):
        raise NotImplementedError

    @abstractmethod
    def get_crs(self):
        raise NotImplementedError

    @abstractmethod
    def get_bbox(self):
        raise NotImplementedError

    def get_geobox(self, bbox):
        snapped_bbox = self.snap_bbox_to_grid(transform(bbox, get_transform(4326, self.get_crs())))
        res = defaultdict(list)
        for b in self.get_assets_as_bands():
            res[GeoBox.from_bbox(snapped_bbox.bounds,
                                 crs=self.get_crs(),
                                 resolution=b.spatial_resolution)].append(b.name)
        return dict(res)

    @abstractmethod
    def load_cube(self, items, bands, geobox, split_by=100, chunks: dict = None):
        raise NotImplementedError

    def get_resampling_per_band(self, target_resolution):
        return {b.name: ("nearest"
                         if target_resolution <= b.spatial_resolution or math.isclose(target_resolution, b.spatial_resolution)
                         else "bilinear"
                         if b.continuous_measurement
                         else "mode") for b in self.get_assets_as_bands()}

    def get_dtype_per_band(self, target_resolution):
        resampling = self.get_resampling_per_band(target_resolution)
        return {b.name: (b.data_type
                         if resampling[b.name] == 'nearest'
                         else "float32")
                for b in self.get_assets_as_bands()}

    def get_band_attributes(self, bands: list[str] | set[str]):
        return {b: self.all_bands[b].to_dict() for b in bands if b in self.all_bands}

    def solarday_offset_seconds(self, item):
        item_centroid = self.__class__(item).get_bbox().centroid
        item_centroid = transform(item_centroid, get_transform(self.get_crs(), 4326))
        longitude = item_centroid.x
        return int(longitude / 15) * 3600

    def sort_items_by_datetime(self, items):
        return sorted(items, key=lambda x: x.properties[self.datetime_id])

    def split_items_keep_solar_days_together(self, items, split_by):
        """
        Split items by solar day.
        Args:
            items (list): List of items to split.
        Returns:
            dict: Dictionary with solar days as keys and lists of items as values.
        """
        rounded_datetimes = []
        for item in items:
            # modify the datetime here to represent solar daytime (offset by time * longitude)
            dt = datetime.fromisoformat(item.properties[self.datetime_id].replace('Z', ''))
            dt += timedelta(seconds=self.solarday_offset_seconds(item))
            rounded_datetimes.append(dt.replace(hour=0, minute=0, second=0, microsecond=0))

        sorted_datetimes = sorted(rounded_datetimes)
        ranks = [sorted_datetimes.index(dt) + 1 for dt in rounded_datetimes]

        # Group elements by rank
        rank_groups = defaultdict(list)
        for index, rank in enumerate(ranks):
            rank_groups[rank].append(index)

        # Convert the rank groups to a list of lists
        ranked_elements = list(rank_groups.values())

        batches = self.split_into_batches(ranked_elements, split_by)

        # Output the batches
        split_items = []
        for batch in batches:
            split_items.append([items[i] for i in batch])
        return split_items

    @staticmethod
    def split_into_batches(elements, batch_size):
        batches = []
        current_batch = []
        current_size = 0

        for group in elements:
            if current_size + len(group) > batch_size:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            current_batch.extend(group)
            current_size += len(group)

        if current_batch:
            batches.append(current_batch)

        return batches

    def centroid_distance_to(self, shape, shape_crs=4326):
        """
        Calculate the distance from the item's centroid to a given shape within the crs of the item.
        Args:
            shape (Polygon): The shape to calculate the distance to.
        Returns:
            float: The distance from the item's centroid to the shape.
        """
        item_centroid = self.get_bbox().centroid
        transformed_shape = transform(shape, get_transform(shape_crs, self.get_crs()))
        return item_centroid.distance(transformed_shape.centroid)


class ASFResultProcessor(ABC):
    # Class-level info — override in subclass
    tilename: str = None
    supported_bands: list = []

    dataset = None
    processingLevel = None
    platform = None
    gridded = False
    overlap = False
    cubing = 'preferred'
    provider = ASFProvider()
    x = 'x'
    y = 'y'
    datetime_id = ''
    all_bands = {}

    def __init__(self, item: Item | Products.OPERAS1Product):
        self.item = item
        if isinstance(self.item, Item):
            self.collection = item.collection_id
        elif isinstance(self.item, Products.OPERAS1Product):
            self.collection = item.get_classname()

    @classmethod
    @abstractmethod
    def download_tiles_to_file(cls, path, items, bands, processes=4):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def generate_stac_items(cls, items):
        raise NotImplementedError

    @classmethod
    def get_supported_bands(cls):
        return cls.supported_bands

    @classmethod
    def get_tilename_key(cls):
        return cls.tilename

    @classmethod
    def request_items(cls, _, request_time, request_place, max_items=None, **kwargs):
        if 'maxResults' in kwargs:
            if max_items is None:
                max_items = kwargs['maxResults']
            del kwargs['maxResults']
        return cls.provider.request_items(
            request_time,
            request_place,
            dataset=cls.dataset,
            processingLevel=cls.processingLevel,
            platform=cls.platform,
            maxResults=max_items,
            **kwargs,
        )

    @abstractmethod
    def snap_bbox_to_grid(self, bbox):
        raise NotImplementedError

    def get_assets_as_bands(self):
        return self.all_bands.values()

    def does_cover_data(self, request_box, input_crs=None):
        if input_crs is not None:
            request_box = transform(request_box, get_transform(input_crs, self.get_crs()))
        return self.get_data_coverage_geometry().contains_properly(request_box)

    @abstractmethod
    def get_data_coverage_geometry(self):
        raise NotImplementedError

    @abstractmethod
    def get_crs(self):
        raise NotImplementedError

    @abstractmethod
    def get_bbox(self):
        raise NotImplementedError

    def get_geobox(self, bbox):
        snapped_bbox = self.snap_bbox_to_grid(transform(bbox, get_transform(4326, self.get_crs())))
        res = defaultdict(list)
        for b in self.get_assets_as_bands():
            res[GeoBox.from_bbox(snapped_bbox.bounds,
                                 crs=self.get_crs(),
                                 resolution=b.spatial_resolution)].append(b.name)
        return dict(res)

    @abstractmethod
    def load_cube(self, items, bands, geobox, split_by=100, chunks: dict = None):
        raise NotImplementedError

    def get_resampling_per_band(self, target_resolution):
        return {b.name: ("nearest"
                         if target_resolution <= b.spatial_resolution or math.isclose(target_resolution, b.spatial_resolution)
                         else "bilinear"
                         if b.continuous_measurement
                         else "mode") for b in self.get_assets_as_bands()}

    def get_dtype_per_band(self, target_resolution):
        resampling = self.get_resampling_per_band(target_resolution)
        return {b.name: (b.data_type
                         if resampling[b.name] == 'nearest'
                         else "float32")
                for b in self.get_assets_as_bands()}

    def get_band_attributes(self, bands: list[str] | set[str]):
        return {b: self.all_bands[b].to_dict() for b in bands if b in self.all_bands}

    def solarday_offset_seconds(self, item):
        item_centroid = self.__class__(item).get_bbox().centroid
        item_centroid = transform(item_centroid, get_transform(self.get_crs(), 4326))
        longitude = item_centroid.x
        return int(longitude / 15) * 3600

    def split_items_keep_solar_days_together(self, items, split_by):
        """
        Split items by solar day.
        Args:
            items (list): List of items to split.
        Returns:
            dict: Dictionary with solar days as keys and lists of items as values.
        """
        rounded_datetimes = []
        for item in items:
            # modify the datetime here to represent solar daytime (offset by time * longitude)
            dt = datetime.fromisoformat(item.properties[self.datetime_id].replace('Z', ''))
            dt += timedelta(seconds=self.solarday_offset_seconds(item))
            rounded_datetimes.append(dt.replace(hour=0, minute=0, second=0, microsecond=0))

        sorted_datetimes = sorted(rounded_datetimes)
        ranks = [sorted_datetimes.index(dt) + 1 for dt in rounded_datetimes]

        # Group elements by rank
        rank_groups = defaultdict(list)
        for index, rank in enumerate(ranks):
            rank_groups[rank].append(index)

        # Convert the rank groups to a list of lists
        ranked_elements = list(rank_groups.values())

        batches = self.split_into_batches(ranked_elements, split_by)

        # Output the batches
        split_items = []
        for batch in batches:
            split_items.append([items[i] for i in batch])
        return split_items

    @staticmethod
    def split_into_batches(elements, batch_size):
        batches = []
        current_batch = []
        current_size = 0

        for group in elements:
            if current_size + len(group) > batch_size:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            current_batch.extend(group)
            current_size += len(group)

        if current_batch:
            batches.append(current_batch)

        return batches


class STACItemProcessor(ABC):
    # Class-level info — override in subclass
    tilename: str = None
    supported_bands: list = []
    overlap = False
    cubing = 'native'
    provider = STACProvider()
    x = 'x'
    y = 'y'
    datetime_id = ''
    all_bands = {}

    def __init__(self, item: Item):
        self.item = item
        self.collection = item.collection_id

    @classmethod
    def download_tiles_to_file(cls, path, items, bands):
        dl_items = []
        for i in items:
            assets = i.get_assets()
            selected_assets = [assets.get(band, None).href for band in bands]
            selected_assets = [(s, path / s.split('/')[-1]) for s in selected_assets]
            dl_items.extend(selected_assets)
        cls.provider.download_granules_to_file(dl_items)

    @classmethod
    def get_supported_bands(cls):
        return cls.supported_bands

    @classmethod
    def get_tilename_key(cls):
        return cls.tilename

    @classmethod
    def request_items(cls, collection, request_time, request_place, item_limit=None):
        items = cls.provider.request_items(collection=collection,
                                           request_time=request_time,
                                           request_place=request_place,
                                           max_items=item_limit,
                                           max_retry=5)
        return items

    @classmethod
    def request_items_tile(cls, collection, request_time, tile_key, tile_ids, **kwargs):
        items = cls.provider.request_items(collection=collection,
                                           request_time=request_time,
                                           query={tile_key: {'in': tile_ids}},
                                           **kwargs)
        return items

    def get_band_attributes(self, bands: list[str] | set[str]):
        return {b: self.all_bands[b].to_dict() for b in bands if b in self.all_bands}

    def get_bbox(self):
        return box(*self.item.bbox)

    def get_crs(self):
        return int(self.item.properties['proj:code'].split(':')[-1])

    @abstractmethod
    def snap_bbox_to_grid(self, bbox):
        raise NotImplementedError

    # get_data_asset_keys

    def get_data_coverage_geometry(self):
        return self.get_bbox()

    # @abstractmethod
    # def get_datetime_property_id(self):
    #     raise NotImplementedError

    def centroid_distance_to(self, shape, shape_crs=4326):
        """
        Calculate the distance from the item's centroid to a given shape within the crs of the item.
        Args:
            shape (Polygon): The shape to calculate the distance to.
        Returns:
            float: The distance from the item's centroid to the shape.
        """
        item_centroid = self.get_bbox().centroid
        transformed_shape = transform(shape, get_transform(shape_crs, self.get_crs()))
        return item_centroid.distance(transformed_shape.centroid)

    def contains_shape(self, shape, shape_crs=4326):
        """
        Check if the item contains a given shape  within the crs of the item.
        Args:
            shape (Polygon): The shape to check against the item's bounding box.
        Returns:
            bool: True if the item contains the shape, False otherwise.
        """
        bbox = self.get_bbox()
        transformed_shape = transform(shape, get_transform(shape_crs, self.get_crs()))
        return bbox.contains(transformed_shape)

    def does_cover_data(self, request_box, input_crs=None):
        if input_crs is not None:
            request_box = transform(request_box, get_transform(input_crs, self.get_crs()))
        return self.get_data_coverage_geometry().contains_properly(request_box)

    def get_assets_as_bands(self):
        return self.all_bands.values()

    def get_dtype_per_band(self, target_resolution):
        resampling = self.get_resampling_per_band(target_resolution)
        return {b.name: (b.data_type
                         if resampling[b.name] == 'nearest'
                         else "float32")
                for b in self.get_assets_as_bands()}

    def get_geobox(self, bbox):
        snapped_bbox = self.snap_bbox_to_grid(transform(bbox, get_transform(4326, self.get_crs())))
        res = defaultdict(list)
        for b in self.get_assets_as_bands():
            res[GeoBox.from_bbox(snapped_bbox.bounds,
                                 crs=self.get_crs(),
                                 resolution=b.spatial_resolution)].append(b.name)
        return dict(res)

    def get_resampling_per_band(self, target_resolution):
        return {b.name: ("nearest"
                         if target_resolution <= b.spatial_resolution or math.isclose(target_resolution, b.spatial_resolution)
                         else "bilinear"
                         if b.continuous_measurement
                         else "mode") for b in self.get_assets_as_bands()}

    def get_tilename_value(self):
        if self.tilename is None:
            raise ValueError("Tilename is not defined for this class.")
        return self.item.properties[self.tilename]

    def get_transform(self, from_crs=None, to_crs=None):
        assert to_crs or from_crs, "define one of to_crs or from_crs"
        return get_transform(from_crs=from_crs or self.get_crs(), to_crs=to_crs or self.get_crs())

    def overlap_percentage(self, shape, shape_crs=4326):
        """
        Calculate the percentage of overlap between the item and a given shape within the crs of the item.
        Args:
            shape (Polygon): The shape to calculate the overlap with.
        Returns:
            float: The percentage of overlap between the item and the shape.
        """
        item_bbox = self.get_bbox()
        transformed_shape = transform(shape, get_transform(shape_crs, self.get_crs()))
        intersection = item_bbox.intersection(transformed_shape)
        return intersection.area / transformed_shape.area

    def solarday_offset_seconds(self, item):
        item_centroid = self.__class__(item).get_bbox().centroid
        item_centroid = transform(item_centroid, get_transform(self.get_crs(), 4326))
        longitude = item_centroid.x
        return int(longitude / 15) * 3600

    def sort_items_by_datetime(self, items):
        return sorted(items, key=lambda x: x.properties[self.datetime_id])

    def split_items_keep_solar_days_together(self, items, split_by):
        """
        Split items by solar day.
        Args:
            items (list): List of items to split.
        Returns:
            dict: Dictionary with solar days as keys and lists of items as values.
        """
        rounded_datetimes = []
        for item in items:
            # modify the datetime here to represent solar daytime (offset by time * longitude)
            dt = datetime.fromisoformat(item.properties[self.datetime_id].replace('Z', ''))
            dt += timedelta(seconds=self.solarday_offset_seconds(item))
            rounded_datetimes.append(dt.replace(hour=0, minute=0, second=0, microsecond=0))

        sorted_datetimes = sorted(rounded_datetimes)
        ranks = [sorted_datetimes.index(dt) + 1 for dt in rounded_datetimes]

        # Group elements by rank
        rank_groups = defaultdict(list)
        for index, rank in enumerate(ranks):
            rank_groups[rank].append(index)

        # Convert the rank groups to a list of lists
        ranked_elements = list(rank_groups.values())

        batches = self.split_into_batches(ranked_elements, split_by)

        # Output the batches
        split_items = []
        for batch in batches:
            split_items.append([items[i] for i in batch])
        return split_items

    def load_cube(self, items, bands, geobox, split_by=100, chunks: dict = None):
        if len(items) > 100:
            logging.warning('large amount of assets, consider loading split in smaller time steps!')

        parameters = {
            "groupby": "solar_day",
            "fail_on_error": True,
        }

        if chunks:
            assert set(chunks.keys()) == {"time", self.x, self.y}, f"Chunks must contain the dimensions 'time', {self.x}, {self.y}!"
            parameters['chunks'] = chunks

        multires_cube = {}
        for gb, band_subset in geobox.items():
            req_bands = set(band_subset) & set(bands)
            if len(req_bands) == 0:
                logging.warning(f'no bands found for {band_subset} in {bands}')
                continue
            resampling = self.get_resampling_per_band(gb.resolution.x)
            dtypes = self.get_dtype_per_band(gb.resolution.x)
            resampling = {k: resampling[k] for k in req_bands if k in resampling}
            dtypes = {k: dtypes[k] for k in req_bands if k in dtypes}

            parameters['bands'] = req_bands
            parameters['geobox'] = gb
            parameters['resampling'] = resampling
            parameters['dtype'] = dtypes
            if split_by is not None and len(items) > split_by:
                split_items = self.split_items_keep_solar_days_together(items, split_by)
                cube = []
                for split in split_items:
                    parameters['items'] = split
                    cube.append(self.provider.download_cube(parameters))
                cube = xr.concat(cube, dim="time")

            else:
                parameters['items'] = items
                cube = self.provider.download_cube(parameters)
            attrs = self.get_band_attributes(req_bands)
            for band in cube.keys():
                cube[band].attrs = attrs[band]

            multires_cube[int(gb.resolution.x)] = cube

        coarsest = max(multires_cube.keys())
        first_var = list(multires_cube[coarsest].data_vars.keys())[0]
        mean_over_time = multires_cube[coarsest][first_var].mean(dim=[self.x, self.y])
        na_value = multires_cube[coarsest][first_var].attrs['_FillValue']
        mask_over_time = np.where(mean_over_time != na_value)[0]

        chunking = {"time": 2 if not chunks else chunks['time']}
        for spat_res in multires_cube.keys():
            # remove empty images, could be moved into separate function
            multires_cube[spat_res] = multires_cube[spat_res].isel(time=mask_over_time)

            chunking[f'x{spat_res}'] = -1 if not chunks else chunks[self.x]
            chunking[f'y{spat_res}'] = -1 if not chunks else chunks[self.y]
            multires_cube[spat_res] = multires_cube[spat_res].rename({self.x: f'x{spat_res}', self.y: f'y{spat_res}'})

        multires_cube = xr.merge(multires_cube.values())

        multires_cube = multires_cube.chunk(chunking)

        return multires_cube

    @staticmethod
    def split_into_batches(elements, batch_size):
        batches = []
        current_batch = []
        current_size = 0

        for group in elements:
            if current_size + len(group) > batch_size:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            current_batch.extend(group)
            current_size += len(group)

        if current_batch:
            batches.append(current_batch)

        return batches


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
            "dtype": self.data_type,
            "_FillValue": self.nodata_value,
            "Spatial Resolution": self.spatial_resolution,
            "Continuous Measurement": self.continuous_measurement,
        }
        # Merge extra attributes into the dictionary
        return {**base_dict, **self.extra_attributes}

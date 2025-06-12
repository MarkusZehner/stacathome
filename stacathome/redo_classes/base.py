from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Any

import logging
import math
from datetime import datetime, timedelta
from pystac import Item
import pystac
from odc.geo.geobox import GeoBox
import numpy as np
from pystac.utils import str_to_datetime
import xarray as xr
from shapely import transform, box  # , Polygon
import rasterio
from rasterio.env import Env
from asf_search import Products

# from rio_stac.stac import PROJECTION_EXT_VERSION, RASTER_EXT_VERSION, EO_EXT_VERSION
from rio_stac.stac import (
    get_dataset_geom,
    get_projection_info,
    get_raster_info,
    bbox_to_geom,
)

from stacathome.redo_classes.generic_utils import get_transform
from stacathome.redo_classes.providers import STACProvider, ASFProvider


logging.basicConfig(
    level=logging.INFO,  # Set to WARNING or ERROR in production
    format="%(asctime)s - %(levelname)s - %(message)s"
)


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

        attrs_from_results = ['flightDirection', 'pathNumber',
                              'processingLevel', 'url', 'startTime',
                              'platform', 'orbit', 'sensor',
                              'subswath', 'beamModeType', 'operaBurstID']

        media_type = 'image/tiff'  # we could also use rio_stac.stac.get_media_type

        collection = "OPERAS1Product"

        # extensions = [
        #     f"https://stac-extensions.github.io/projection/{PROJECTION_EXT_VERSION}/schema.json",
        #     f"https://stac-extensions.github.io/raster/{RASTER_EXT_VERSION}/schema.json",
        #     f"https://stac-extensions.github.io/eo/{EO_EXT_VERSION}/schema.json",
        # ]
        return_items = []
        for i, paths in items.items():
            meta_from_item = {a: i.properties[a] for a in attrs_from_results}
            fileID = i.properties['fileID']

            assets = [{
                "name": filename.split('/')[-1].replace('.tif', '').split('v1.0_')[1],
                "path": filename,
                "href": None,
                "role": "data",
            } for filename in paths if filename.endswith('.tif')]

            # for filename in paths:
            #     if filename.endswith('.tif'):
            #         print(filename)
            #         print(filename.split('/')[-1].replace('.tif', ''))

            # exit()
            bboxes = []
            proj_bboxes = []
            pystac_assets = []

            crs = []

            for asset in assets:
                with Env(GTIFF_SRS_SOURCE='EPSG'):  # CRS definition in the GTiff differs from EPSG:
                    # WARNING:rasterio._env:CPLE_AppDefined in The definition of projected CRS EPSG:32653 got from GeoTIFF
                    # keys is not the same as the one from the EPSG registry, which may cause issues during reprojection operations.
                    # Set GTIFF_SRS_SOURCE configuration option to EPSG to use official parameters (overriding the ones from GeoTIFF keys),
                    # or to GEOKEYS to use custom values from GeoTIFF keys and drop the EPSG code.

                    # choosing between the two options gives different values in the image position after the 9th decimal place (lat lon)
                    with rasterio.open(asset["path"]) as src_dst:
                        # Get BBOX and Footprint
                        crs_proj = get_projection_info(src_dst)['epsg']
                        crs.append(crs_proj)
                        proj_geom = get_dataset_geom(src_dst, densify_pts=0, precision=-1, geographic_crs=crs_proj)
                        proj_bboxes.append(proj_geom["bbox"])
                        # stac items geometry and bbox need to be in lat lon
                        dataset_geom = get_dataset_geom(src_dst, densify_pts=0, precision=-1)  # , geographic_crs=crs_proj)
                        bboxes.append(dataset_geom["bbox"])

                        proj_info = {
                            f"proj:{name}": value
                            for name, value in get_projection_info(src_dst).items() if name in ['bbox', 'shape', 'transform']
                        }

                        raster_info = {
                            "raster:bands": get_raster_info(src_dst, max_size=1024)
                        }

                        pystac_assets.append(
                            (
                                asset["name"],
                                pystac.Asset(
                                    href=asset["href"] or src_dst.name,
                                    media_type=media_type,
                                    extra_fields={
                                        **proj_info,
                                        **raster_info,
                                    },
                                    roles=asset["role"],
                                ),
                            )
                        )

            minx, miny, maxx, maxy = zip(*proj_bboxes)
            proj_bbox = [min(minx), min(miny), max(maxx), max(maxy)]

            minx, miny, maxx, maxy = zip(*bboxes)
            bbox = [min(minx), min(miny), max(maxx), max(maxy)]

            if len(set(crs)) > 1:
                raise ValueError("Multiple CRS found in the assets")

            meta_from_item['proj:code'] = crs[0]
            meta_from_item['proj:bbox'] = proj_bbox

            # item
            item = pystac.Item(
                id=fileID,
                geometry=bbox_to_geom(bbox),
                bbox=bbox,
                collection=collection,
                # stac_extensions=extensions,
                datetime=str_to_datetime(meta_from_item['startTime']),
                properties=meta_from_item,
            )

            # if we add a collection we MUST add a link
            if collection:
                item.add_link(
                    pystac.Link(
                        pystac.RelType.COLLECTION,
                        meta_from_item['url'] or collection,
                        media_type=pystac.MediaType.JSON,
                    )
                )
            for key, asset in pystac_assets:
                item.add_asset(key=key, asset=asset)

            return_items.append(item)

        return return_items

    @classmethod
    def get_supported_bands(cls):
        return cls.supported_bands

    @classmethod
    def get_tilename_key(cls):
        return cls.tilename

    @classmethod
    def request_items(cls, _, request_time, request_place, max_items=None, **kwargs):
        if 'maxResults' in kwargs:
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
    def snap_bbox_to_grid(cls):
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
        # if isinstance(self.item, Item):
        #     Polygon(self.item.properties.geometry['coordinates'][0])
        # elif isinstance(self.item, Products.OPERAS1Product):
        #     return Polygon(self.item.geometry['coordinates'][0])

    @abstractmethod
    def get_crs(self):
        raise NotImplementedError
        # if isinstance(self.item, Item):
        #     return self.item.properties['proj:code']
        # elif isinstance(self.item, Products.OPERAS1Product):
        #     return 4236

    @abstractmethod
    def get_bbox(self):
        raise NotImplementedError
        # if isinstance(self.item, Item):
        #     return box(*self.item.bbox)
        # elif isinstance(self.item, Products.OPERAS1Product):
        #     return box(*Polygon(self.item.geometry['coordinates'][0]).bounds)

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


class STACItemProcessor(ABC):
    # Class-level info — override in subclass
    tilename: str = None
    supported_bands: list = []
    overlap = False
    cubing = 'native'
    provider = STACProvider()
    x = 'x'
    y = 'y'

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

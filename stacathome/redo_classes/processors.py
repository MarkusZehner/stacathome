import os
import re
import copy
from collections import defaultdict
from urllib import parse
import math
import datetime
import numpy as np
import xarray as xr
from shapely import transform, Polygon, box
from pyproj import CRS
import asf_search
from asf_search import Products
from earthaccess.results import DataGranule

from asf_search.download.file_download_type import FileDownloadType
# from asf_search.download import download_urls
import pystac
from pystac import Item
from pystac.utils import str_to_datetime
import rasterio
from rasterio.env import Env

# from rio_stac.stac import PROJECTION_EXT_VERSION, RASTER_EXT_VERSION, EO_EXT_VERSION
from rio_stac.stac import (
    get_dataset_geom,
    get_projection_info,
    get_raster_info,
    bbox_to_geom,
)

import logging

from stacathome.redo_classes.base import STACItemProcessor, ASFResultProcessor, Band, EarthAccessProcessor
from stacathome.redo_classes.generic_utils import (get_transform, compute_scale_and_offset, create_utm_grid_bbox, arange_bounds,
                                                   most_common, resolve_best_containing, merge_to_cover, merge_item_datetime_by_timedelta)

logging.basicConfig(
    level=logging.INFO,  # Set to WARNING or ERROR in production
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ECOL2TLSTEProcessor(EarthAccessProcessor):
    print('could the shift be from 01 or 02 processing iteration???')
    collection = "ECO_L2T_LSTE.002"
    datetime_id = 'startTime'
    cubing = 'custom'

    spatial_res = 70

    float32 = ['view_zenith', 'height', 'LST', 'LST_err', 'EmisWB']
    uint16 = ['QC']
    uint8 = ['water', 'cloud']
    float32_bands = {}
    uint16_bands = {}
    uint8_bands = {}
    for i in float32:
        float32_bands[i] = Band(
            name=i,
            data_type='float32',
            nodata_value=np.nan,
            spatial_resolution=spatial_res,
            continuous_measurement=True,
            valid_range=(0, 1) if i == 'EmisWB' else None,
        )
    for i in uint16:
        uint16_bands[i] = Band(
            name=i,
            data_type='uint16',
            nodata_value=None,
            spatial_resolution=spatial_res,
            continuous_measurement=False,
        )
    for i in uint8:
        uint8_bands[i] = Band(
            name=i,
            data_type='uint8',
            nodata_value=None,
            spatial_resolution=spatial_res,
            continuous_measurement=False,
            valid_range=(0, 1),
        )

    all_bands = float32_bands | uint16_bands | uint8_bands
    supported_bands = list(all_bands.keys())

    @classmethod
    def download_tiles_to_file(cls, path, items, bands=None, processes=4):
        granules_links = {g: g.data_links() for g in items}
        granules_links = cls.filter_highest_version_links(granules_links)

        if bands:
            bands = [b + '.tif' for b in bands]
            for key in granules_links.keys():
                granules_links[key] = [s for s in granules_links[key] if s.endswith(tuple(bands))]

        granules_links_flat = [l for v in granules_links.values() for l in v]

        cls.provider.download_from_earthaccess(granules_links_flat, path, threads=processes)

        for k in granules_links.keys():
            granules_links[k] = [os.path.join(path, os.path.split(parse.urlparse(paths).path)[1]) for paths in granules_links[k]]

        return granules_links

    @classmethod
    def generate_stac_items(cls, items):
        collection = "ECO_L2T_LSTE.002"
        media_type = 'image/tiff'

        return_items = []

        for i, paths in items.items():
            meta_from_item = {}
            fileID = i['meta']['native-id']
            meta_from_item['provider-id'] = i['meta']['provider-id']
            meta_from_item['startTime'] = i['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
            meta_from_item['BeginOrbitNumber'] = i['umm']['OrbitCalculatedSpatialDomains'][0]['BeginOrbitNumber']
            meta_from_item['DayNightFlag'] = i['umm']['DataGranule']['DayNightFlag']

            assets = [{
                "name": re.split(r'_(\d{2})_', filename)[-1].replace('.tif', ''),
                "path": filename,
                "href": None,
                "role": "data",
            } for filename in paths if filename.endswith('.tif')]

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
            meta_from_item['url'] = None

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

        return return_items[::-1]

    def get_bbox(self):
        if isinstance(self.item, Item):
            return box(*self.item.properties["proj:bbox"])
        elif isinstance(self.item, DataGranule):
            bounds = self.item['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['BoundingRectangles'][0]
            xmin = bounds['WestBoundingCoordinate']
            xmax = bounds['EastBoundingCoordinate']
            ymin = bounds['SouthBoundingCoordinate']
            ymax = bounds['NorthBoundingCoordinate']
            return box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    def get_crs(self):
        if isinstance(self.item, Item):
            return self.item.properties['proj:code']
        elif isinstance(self.item, DataGranule):
            return 4236

    def snap_bbox_to_grid(self, bbox, grid_size=70):
        inherent_bbox = self.get_bbox()
        inherent_x, inherent_y = arange_bounds(inherent_bbox.bounds, grid_size)
        created_box = create_utm_grid_bbox(bbox.bounds, grid_size, 40, -40)
        created_x, created_y = arange_bounds(created_box.bounds, grid_size)

        print(created_x[:5], inherent_x[:2], inherent_x[-2:])
        print(created_y[:5], inherent_y[:2], inherent_y[-2:])

        # Determine the overlapping grid coordinates
        shared_x = set(created_x) & set(inherent_x)
        shared_y = set(created_y) & set(inherent_y)

        if len(created_x) >= len(inherent_x):
            overlap_x = created_x[(created_x >= min(inherent_x)) & (created_x <= max(inherent_x))]
            overlap_y = created_y[(created_y >= min(inherent_y)) & (created_y <= max(inherent_y))]

        else:
            overlap_x = inherent_x[(inherent_x >= min(created_x)) & (inherent_x <= max(created_x))]
            overlap_y = inherent_y[(inherent_y >= min(created_y)) & (inherent_y <= max(created_y))]

        if len(shared_x) != len(overlap_x) or len(shared_y) != len(overlap_y):
            raise ValueError("Grid of created box does not fully align with inherent grid")

        return created_box

    def get_data_coverage_geometry(self):
        return self.get_bbox()

    def load_cube(self, items, bands, geobox, split_by=100, chunks: dict = None):
        raise NotImplementedError('Ecostress L2T.002 LSTE has observed shifts in x, therefore custom cubing to remedy!')
        if 'height' in bands or 'view_zenith' in bands:
            logging.info('the current method does load static assets for each time step, for long time series consider not using height or view_zenith bands!')
        if len(items) > 100:
            logging.warning('large amount of assets, consider loading split in smaller time steps!')

        items = self.sort_items_by_datetime(items)
        print('after sorting, before cubing')
        print(items[0].properties[self.datetime_id])
        print(items[-1].properties[self.datetime_id])
        # items = merge_item_datetime_by_timedelta(items)

        parameters = {
            "groupby": "id",
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
                    cube.append(self.provider.create_cube(parameters))
                cube = xr.concat(cube, dim="time")

            else:
                parameters['items'] = items
                cube = self.provider.create_cube(parameters)
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


class OPERASentinel1RTCProcessor(ASFResultProcessor):
    dataset = "OPERA-S1"
    processingLevel = "RTC"
    platform = asf_search.PLATFORM.SENTINEL1
    gridded = True
    datetime_id = 'startTime'

    special_bands = {
        "mask" :
            Band(
                name="mask",
                data_type="uint8",
                nodata_value=255,
                spatial_resolution=30,
                continuous_measurement=False,
                flag_meanings=[
                    "Valid sample not affected by layover or shadow",
                    "Valid sample affected by shadow",
                    "Valid sample affected by layover",
                    "Valid sample affected by layover and shadow",
                    "Invalid sample (fill value)",
                ],
                flag_values=[0, 1, 2, 3, 255],
            ),
    }
    bands = {
        "VV" :
            Band(
                name="VV",
                data_type="float32",
                nodata_value=np.nan,
                spatial_resolution=30,
                continuous_measurement=True,
            ),
        "VH" :
            Band(
                name="VH",
                data_type="float32",
                nodata_value=np.nan,
                spatial_resolution=30,
                continuous_measurement=True,
            ),
        # "iso" :
        #     Band(
        #         name="iso",
        #         data_type="float32",
        #         nodata_value=-32768,
        #         spatial_resolution=30,
        #         continuous_measurement=False,
        #     ),
        "incidence_angle" :
            Band(
                name="incidence_angle",
                data_type="float32",
                nodata_value=np.nan,
                spatial_resolution=30,
                continuous_measurement=True,
            ),
        "local_incidence_angle" :
            Band(
                name="local_incidence_angle",
                data_type="float32",
                nodata_value=np.nan,
                spatial_resolution=30,
                continuous_measurement=True,
            ),
        # "static-mask" :
        #     Band(
        #         name="static-mask",
        #         data_type="float32",
        #         nodata_value=-32768,
        #         spatial_resolution=30,
        #         continuous_measurement=False,
        #     ),
        "number_of_looks":
            Band(
                name="number_of_looks",
                data_type="float32",
                nodata_value=np.nan,
                spatial_resolution=30,
                continuous_measurement=True,
            ),
        "rtc_anf_gamma0_to_beta0":
            Band(
                name="rtc_anf_gamma0_to_beta0",
                data_type="float32",
                nodata_value=np.nan,
                spatial_resolution=30,
                continuous_measurement=True,
            ),
        "rtc_anf_gamma0_to_sigma0":
            Band(
                name="rtc_anf_gamma0_to_sigma0",
                data_type="float32",
                nodata_value=np.nan,
                spatial_resolution=30,
                continuous_measurement=True,
            ),
        # "static-iso" :
        #     Band(
        #         name="static-iso",
        #         data_type="float32",
        #         nodata_value=-32768,
        #         spatial_resolution=30,
        #         continuous_measurement=False,
        #     ),
    }

    all_bands = bands | special_bands
    supported_bands = list(all_bands.keys())

    @classmethod
    def download_tiles_to_file(cls, path, items, bands, processes=4):
        dl_items = {}
        bands = list(set(cls.supported_bands) & set(bands))
        static_pre = 'https://datapool.asf.alaska.edu/RTC-STATIC/OPERA-S1/OPERA_L2_RTC-S1-STATIC_'
        static_post = '_20140403_S1A_30_v1.0'
        fileType = FileDownloadType.ALL_FILES

        for i in items:
            urls = []
            dynamic_urls = i.get_urls(fileType)

            for u in dynamic_urls:
                for b in bands:
                    if b in u:
                        urls.append(u)

            for b in bands:
                if b not in ['VV', 'VH', 'mask']:
                    file_type = '.tif'  # '.xml' if b == 'static-iso' else '.tif'
                    filler = '_'  # '.' if b == 'static-iso' else '_'
                    inc_url = (
                        static_pre +
                        i.properties['operaBurstID'].replace('_', '-') +
                        static_post + filler + b  # .split('-')[1]
                        + file_type)
                    urls.append(inc_url)

            dl_items[i] = urls

        dl_items_flat = [item for sublist in dl_items.values() for item in sublist]
        cls.provider.download_from_asf(dl_items_flat, path=path, processes=processes)

        for k in dl_items.keys():
            dl_items[k] = [os.path.join(path, os.path.split(parse.urlparse(paths).path)[1]) for paths in dl_items[k]]

        # make stac items here? -> better own method to call from?
        # dl_items = cls.generate_stac_items(dl_items)

        return dl_items

    @classmethod
    def generate_stac_items(cls, items):
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

            meta_from_item_startTime = str_to_datetime(meta_from_item['startTime'])

            # del meta_from_item['startTime']

            # item
            item = pystac.Item(
                id=fileID,
                geometry=bbox_to_geom(bbox),
                bbox=bbox,
                collection=collection,
                # stac_extensions=extensions,
                datetime=meta_from_item_startTime,
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

    def get_bbox(self):
        if isinstance(self.item, Item):
            return box(*self.item.properties["proj:bbox"])
        elif isinstance(self.item, Products.OPERAS1Product):
            return box(*Polygon(self.item.geometry['coordinates'][0]).bounds)

    def get_crs(self):
        if isinstance(self.item, Item):
            return self.item.properties['proj:code']
        elif isinstance(self.item, Products.OPERAS1Product):
            return 4236

    def get_data_coverage_geometry(self):
        if isinstance(self.item, Item):
            return transform(Polygon(*self.item.geometry["coordinates"]), get_transform(4326, self.get_crs()))
        elif isinstance(self.item, Products.OPERAS1Product):
            return Polygon(self.item.geometry['coordinates'][0])

    def sort_items_by_datetime(self, items):
        return sorted(items, key=lambda x: x.properties[self.datetime_id])

    def snap_bbox_to_grid(self, bbox, grid_size=30):
        inherent_bbox = self.get_bbox()
        inherent_x, inherent_y = arange_bounds(inherent_bbox.bounds, grid_size)
        created_box = create_utm_grid_bbox(bbox.bounds, grid_size)
        created_x, created_y = arange_bounds(created_box.bounds, grid_size)

        # Determine the overlapping grid coordinates
        shared_x = set(created_x) & set(inherent_x)
        shared_y = set(created_y) & set(inherent_y)

        if len(created_x) >= len(inherent_x):
            overlap_x = created_x[(created_x >= min(inherent_x)) & (created_x <= max(inherent_x))]
            overlap_y = created_y[(created_y >= min(inherent_y)) & (created_y <= max(inherent_y))]

        else:
            overlap_x = inherent_x[(inherent_x >= min(created_x)) & (inherent_x <= max(created_x))]
            overlap_y = inherent_y[(inherent_y >= min(created_y)) & (inherent_y <= max(created_y))]

        if len(shared_x) != len(overlap_x) or len(shared_y) != len(overlap_y):
            raise ValueError("Grid of created box does not fully align with inherent grid")

        return created_box

    def load_cube(self, items, bands, geobox, split_by=100, chunks: dict = None):
        if {'VV', 'VH', 'mask'} < set(bands):
            logging.info('the current method does load static assets for each time step, for long time series just use VV, VH and mask bands!')
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
                    cube.append(self.provider.create_cube(parameters))
                cube = xr.concat(cube, dim="time")

            else:
                parameters['items'] = items
                cube = self.provider.create_cube(parameters)
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


class Sentinel1RTCProcessor(STACItemProcessor):
    datetime_id = "datetime"
    gridded = True

    special_bands = {}
    bands = {
        "vv" :
            Band(
                name="vv",
                data_type="float32",
                nodata_value=-32768,
                spatial_resolution=10,
                continuous_measurement=True,
            ),
        "vh" :
            Band(
                name="vh",
                data_type="float32",
                nodata_value=-32768,
                spatial_resolution=10,
                continuous_measurement=True,
            ),
        # "hh" :
        #     Band(
        #         name="hh",
        #         data_type="float32",
        #         nodata_value=-32768,
        #         spatial_resolution=10,
        #         continuous_measurement=True,
        #     ),
        # "hv" :
        #     Band(
        #         name="hv",
        #         data_type="float32",
        #         nodata_value=-32768,
        #         spatial_resolution=10,
        #         continuous_measurement=True,
        #     ),

    }

    all_bands = bands | special_bands
    supported_bands = list(all_bands.keys())

    def get_bbox(self):
        return box(*self.item.properties["proj:bbox"])

    def get_data_coverage_geometry(self):
        return transform(Polygon(*self.item.geometry["coordinates"]), get_transform(4326, self.get_crs()))

    def snap_bbox_to_grid(self, bbox, grid_size=10):
        inherent_bbox = self.get_bbox()
        inherent_x, inherent_y = arange_bounds(inherent_bbox.bounds, grid_size)
        created_box = create_utm_grid_bbox(bbox.bounds, grid_size)
        created_x, created_y = arange_bounds(created_box.bounds, grid_size)

        # Determine the overlapping grid coordinates
        shared_x = set(created_x) & set(inherent_x)
        shared_y = set(created_y) & set(inherent_y)

        if len(created_x) >= len(inherent_x):
            overlap_x = created_x[(created_x >= min(inherent_x)) & (created_x <= max(inherent_x))]
            overlap_y = created_y[(created_y >= min(inherent_y)) & (created_y <= max(inherent_y))]

        else:
            overlap_x = inherent_x[(inherent_x >= min(created_x)) & (inherent_x <= max(created_x))]
            overlap_y = inherent_y[(inherent_y >= min(created_y)) & (inherent_y <= max(created_y))]

        if len(shared_x) != len(overlap_x) or len(shared_y) != len(overlap_y):
            raise ValueError("Grid of created box does not fully align with inherent grid")

        return created_box


class LandsatC2L2Processor(STACItemProcessor):
    datetime_id = "datetime"
    gridded = True

    supported_bands = [
        "qa",
        "red",
        "blue",
        "drad",
        "emis",
        "emsd",
        "lwir",
        "trad",
        "urad",
        "atran",
        "cdist",
        "green",
        "nir08",
        "lwir11",
        "swir16",
        "swir22",
        "coastal",
        "cloud_qa",
        "qa_pixel",
        "qa_radsat",
        "qa_aerosol",
        "atmos_opacity"
    ]

    special_bands = {
        'qa' :
            Band(
                'qa',
                'int16',
                -9999,
                30,
                False,
                long_name="Surface Temperature Quality Assessment Band",
            ),
        'qa_pixel' :
            Band(
                'qa_pixel',
                'uint16',
                1,
                30,
                False,
                long_name="Pixel Quality Assessment Band",
                flag_meanings=[
                    "Fill",
                    "Clear with lows set",
                    "Dilated cloud over land",
                    "Water with lows set",
                    "Dilated cloud over water",
                    "Mid conf cloud",
                    "Mid conf cloud over water",
                    "High conf cloud",
                    "High conf cloud shadow",
                    "Water with cloud shadow",
                    "Mid conf cloud with shadow",
                    "Mid conf cloud with shadow over water",
                    "High conf cloud with shadow",
                    "High conf cloud with shadow over water",
                    "High conf snow/ice",
                    "High conf cirrus",
                    "Cirrus, mid cloud",
                    "Cirrus, high cloud"
                ],
                flag_values=[1, 21824, 21826, 21888, 21890, 22080,
                             22144, 22280, 23888, 23952, 24088,
                             24216, 24344, 24472, 30048, 54596,
                             54852, 55052],
            ),
        'qa_aerosol' :
            Band(
                'qa_aerosol',
                'uint8',
                1,
                30,
                False,
                long_name="",
            ),
        'qa_radsat' :
            Band(
                'qa_radsat',
                'uint16',
                1,
                30,
                False,
                long_name="",
            ),
    }

    def get_data_asset_keys(self, role=['data', 'saturation', 'water-mask']):
        if isinstance(role, str):
            role = [role]
        key = [i for r in role for i in self.item.get_assets(role=r).keys()]
        assert len(key) > 0, "No data assets found!"
        return key

    def get_assets_as_bands(self):
        present_bands = self.get_data_asset_keys()
        present_supported_bands = sorted(set(self.supported_bands) & set(present_bands))

        bands = []
        for b in present_supported_bands:
            special = self.special_bands.get(b, None)
            if special:
                bands.append(special)
            else:
                info = self.item.assets[b].to_dict()
                eo_bands = info.get('eo:bands')
                raster_bands = info.get('raster:bands')
                if isinstance(eo_bands, list) and len(eo_bands) == 1:
                    eo_bands = eo_bands[0]
                if isinstance(raster_bands, list) and len(raster_bands) == 1:
                    raster_bands = raster_bands[0]
                bands.append(
                    Band(
                        b,
                        raster_bands.get('data_type'),
                        raster_bands.get('nodata'),
                        raster_bands.get('spatial_resolution'),
                        True,
                        scale=raster_bands.get('scale'),
                        offset=raster_bands.get('offset'),
                        long_name=eo_bands.get('description') if eo_bands else raster_bands.get('title'),
                        center_wavelenght=eo_bands.get('center_wavelength') if eo_bands else 'na',
                        full_width_half_max=eo_bands.get('full_width_half_max') if eo_bands else 'na',
                    )
                )
        return bands

    def get_band_attributes(self, bands: list[str] | set[str]):
        info = self.get_assets_as_bands()
        bandattributes = {}
        for i in info:
            d = i.to_dict()
            bandattributes[d['Name']] = d
        bandattributes = {b: bandattributes[b] for b in bands if b in bandattributes}
        return bandattributes

    def get_bbox(self):
        # correct for point and area
        xmin = self.item.properties['proj:transform'][2]
        ymax = self.item.properties['proj:transform'][5]
        xmax = xmin + (self.item.properties['proj:transform'][0] * self.item.properties['proj:shape'][1])
        ymin = ymax + (self.item.properties['proj:transform'][4] * self.item.properties['proj:shape'][0])
        return box(xmin, ymin, xmax, ymax)

    def snap_bbox_to_grid(self, bbox, grid_size=30):
        inherent_bbox = self.get_bbox()
        inherent_x, inherent_y = arange_bounds(inherent_bbox.bounds, grid_size)
        created_box = create_utm_grid_bbox(bbox.bounds, grid_size, 15, -15)
        created_x, created_y = arange_bounds(created_box.bounds, grid_size)

        # Determine the overlapping grid coordinates
        shared_x = set(created_x) & set(inherent_x)
        shared_y = set(created_y) & set(inherent_y)

        if len(created_x) >= len(inherent_x):
            overlap_x = created_x[(created_x >= min(inherent_x)) & (created_x <= max(inherent_x))]
            overlap_y = created_y[(created_y >= min(inherent_y)) & (created_y <= max(inherent_y))]

        else:
            overlap_x = inherent_x[(inherent_x >= min(created_x)) & (inherent_x <= max(created_x))]
            overlap_y = inherent_y[(inherent_y >= min(created_y)) & (inherent_y <= max(created_y))]

        if len(shared_x) != len(overlap_x) or len(shared_y) != len(overlap_y):
            raise ValueError("Grid of created box does not fully align with inherent grid")

        return created_box

    def get_data_coverage_geometry(self):
        return transform(Polygon(*self.item.geometry["coordinates"]), get_transform(4326, self.get_crs()))

    # def get_no_data_per_band(self):
    #     return {b.name: (b.nodata_value)
    #             for b in self.get_assets_as_bands()}

    def load_cube(self, items, bands, geobox, split_by=100, chunks: dict = None):

        # separate here into the different landsat platforms?
        platform_sparated_items = defaultdict(list)

        for i in items:
            platform_sparated_items[i.properties['platform']].append(i)

        ls_platform_cubes = {}
        for platform, items in platform_sparated_items.items():

            if len(items) > 100:
                logging.warning('large amount of assets, consider loading split in smaller time steps!')

            parameters = {
                "groupby": "solar_day",
                "fail_on_error": False,
            }

            if chunks:
                assert set(chunks.keys()) == {"time", "x", "y"}, "Chunks must contain the dimensions 'time', 'x', 'y'!"
                parameters['chunks'] = chunks

            multires_cube = {}
            for gb, band_subset in geobox.items():
                req_bands = set(band_subset) & set(bands)
                if len(req_bands) == 0:
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

            # remove empty images, could be moved into separate function
            # get coarsest resolution first band and check where mean value is 0:
            coarsest = max(multires_cube.keys())
            first_var = list(multires_cube[coarsest].data_vars.keys())[0]
            mean_over_time = multires_cube[coarsest][first_var].mean(dim=['x', 'y'])
            na_value = multires_cube[coarsest][first_var].attrs['_FillValue']
            mask_over_time = np.where(mean_over_time != na_value)[0]

            chunking = {"time": 2 if not chunks else chunks['time']}
            for spat_res in multires_cube.keys():
                # remove empty images, could be moved into separate function
                multires_cube[spat_res] = multires_cube[spat_res].isel(time=mask_over_time)

                chunking[f'x{spat_res}'] = -1 if not chunks else chunks['x']
                chunking[f'y{spat_res}'] = -1 if not chunks else chunks['y']

                multires_cube[spat_res]['x'] = multires_cube[spat_res]['x'] - (int(spat_res) / 2)
                multires_cube[spat_res]['y'] = multires_cube[spat_res]['y'] + (int(spat_res) / 2)
                multires_cube[spat_res] = multires_cube[spat_res].rename({'x': f'x{spat_res}', 'y': f'y{spat_res}'})

            multires_cube = xr.merge(multires_cube.values())
            multires_cube = multires_cube.chunk(chunking)

            ls_platform_cubes[platform] = multires_cube

        return ls_platform_cubes


class Sentinel2L2AProcessor(STACItemProcessor):
    tilename = "s2:mgrs_tile"
    datetime_id = "datetime"
    gridded = True
    overlap = True

    supported_bands = [
        "B01", "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"
    ]
    special_bands = {
        'SCL' :
            Band(
                'SCL',
                'uint8',
                0,
                20,
                False,
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
    }

    def get_assets_as_bands(self):
        present_bands = self.get_data_asset_keys()
        present_supported_bands = sorted(set(self.supported_bands) & set(present_bands))

        bands = []
        for b in present_supported_bands:
            special = self.special_bands.get(b, None)
            if special:
                bands.append(special)
            else:
                info = self.item.assets[b].to_dict()
                eo_bands = info.get('eo:bands')
                if isinstance(eo_bands, list) and len(eo_bands) == 1:
                    eo_bands = eo_bands[0]
                bands.append(
                    Band(
                        b,
                        "uint16",
                        0,
                        info.get('gsd', 0),
                        True,
                        long_name=eo_bands.get('description'),
                        center_wavelenght=eo_bands.get('center_wavelength'),
                        full_width_half_max=eo_bands.get('full_width_half_max'),
                    )
                )

        return bands

    def get_bbox(self):
        key = self.get_data_asset_keys()[0]
        return box(*self.item.assets[key].extra_fields['proj:bbox'])

    def get_band_attributes(self, bands: list[str] | set[str]):
        info = self.get_assets_as_bands()
        bandattributes = {}
        for i in info:
            d = i.to_dict()
            bandattributes[d['Name']] = d
        bandattributes = {b: bandattributes[b] for b in bands if b in bandattributes}
        return bandattributes

    def get_data_asset_keys(self, role="data"):
        key = list(self.item.get_assets(role=role).keys())
        assert len(key) > 0, "No data assets found!"
        return key

    def get_data_coverage_geometry(self):
        return transform(Polygon(*self.item.geometry["coordinates"]), get_transform(4326, self.get_crs()))

    def collect_covering_tiles_and_coverage(self, request_place, item_limit=12, items=None, request_time=None):
        # move this into the processor classes?
        tile_ids_per_collection = {}
        # for collection in self.collections:
        # if isinstance(self.request_time, dict):
        #     req_time = self.request_time.get(collection, None)
        #     if req_time is None:
        #         logging.warning(f"No time found for {collection} in {self.request_place}")
        #         continue
        # else:
        #     req_time = self.request_time

        if items is None and all([request_time is not None, request_place is not None]):
            items = self.provider.request_items(
                self.item.collection_id,
                request_time=request_time,
                request_place=request_place,
                max_items=item_limit)

            if items is None:
                raise ValueError("No items found for the given request parameters.")
        else:
            filter = True

        if len(items) < item_limit:
            logging.warning(f"Less than {item_limit} items found for {self.item.collection_id} in {request_place} "
                            f"and {request_time}")

        collect_coverage_from = items[:min(len(items), item_limit)]

        by_tile = defaultdict(list)

        for i in collect_coverage_from:
            item = self.__class__(i)
            by_tile[item.get_tilename_value()].append([
                item.get_crs(),
                item.contains_shape(request_place),
                item.centroid_distance_to(request_place),
                item.overlap_percentage(request_place),
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
            found_tiles = merge_to_cover(by_tile_filtered, request_place)

        tile_ids = [t[0] for t in found_tiles]

        # get one item with the same tile_id for the geobox
        item = next(i
                    for i in collect_coverage_from
                    if self.__class__(i).get_tilename_value() == tile_ids[0])

        geobox = self.__class__(item).get_geobox(request_place)

        if filter:
            filtered_items = [i for i in items if self.__class__(i).get_tilename_value() in tile_ids]

        tile_ids_per_collection = {'tile_id': tile_ids, 'geobox': geobox}

        if filter:
            return [filtered_items, tile_ids_per_collection]
        else:
            return tile_ids_per_collection

    def snap_bbox_to_grid(self, bbox, grid_size=60):
        inherent_bbox = self.get_bbox()
        inherent_x, inherent_y = arange_bounds(inherent_bbox.bounds, grid_size)
        created_box = create_utm_grid_bbox(bbox.bounds, grid_size)
        created_x, created_y = arange_bounds(created_box.bounds, grid_size)

        # Determine the overlapping grid coordinates
        shared_x = set(created_x) & set(inherent_x)
        shared_y = set(created_y) & set(inherent_y)

        if len(created_x) >= len(inherent_x):
            overlap_x = created_x[(created_x >= min(inherent_x)) & (created_x <= max(inherent_x))]
            overlap_y = created_y[(created_y >= min(inherent_y)) & (created_y <= max(inherent_y))]

        else:
            overlap_x = inherent_x[(inherent_x >= min(created_x)) & (inherent_x <= max(created_x))]
            overlap_y = inherent_y[(inherent_y >= min(created_y)) & (inherent_y <= max(created_y))]

        if len(shared_x) != len(overlap_x) or len(shared_y) != len(overlap_y):
            raise ValueError("Grid of created box does not fully align with inherent grid")

        return created_box

    def load_cube(self, items, bands, geobox, split_by=100, chunks: dict = None):
        # most of this until the s2 specific harmonization could be moved to the base class

        # implement this in multiprocess?

        if len(items) > 100:
            logging.warning('large amount of assets, consider loading split in smaller time steps!')

        parameters = {
            "groupby": "solar_day",
            "fail_on_error": False,
        }

        if chunks:
            assert set(chunks.keys()) == {"time", "x", "y"}, "Chunks must contain the dimensions 'time', 'x', 'y'!"
            parameters['chunks'] = chunks

        multires_cube = {}
        for gb, band_subset in geobox.items():
            req_bands = set(band_subset) & set(bands)
            if len(req_bands) == 0:
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

        # remove empty images, could be moved into separate function
        # get coarsest resolution first band and check where mean value is 0:
        coarsest = max(multires_cube.keys())
        first_var = list(multires_cube[coarsest].data_vars.keys())[0]
        mean_over_time = multires_cube[coarsest][first_var].mean(dim=['x', 'y'])
        na_value = multires_cube[coarsest][first_var].attrs['_FillValue']
        mask_over_time = np.where(mean_over_time != na_value)[0]

        chunking = {"time": 2 if not chunks else chunks['time']}
        for spat_res in multires_cube.keys():
            # remove empty images, could be moved into separate function
            multires_cube[spat_res] = multires_cube[spat_res].isel(time=mask_over_time)

            chunking[f'x{spat_res}'] = -1 if not chunks else chunks['x']
            chunking[f'y{spat_res}'] = -1 if not chunks else chunks['y']
            multires_cube[spat_res] = multires_cube[spat_res].rename({'x': f'x{spat_res}', 'y': f'y{spat_res}'})

        multires_cube = xr.merge(multires_cube.values())

        # mask by SCL?
        # remove values time steps with no data per SCL layer?

        # this changes int16 to float32 could also be seen as different function
        multires_cube = self.harmonize_s2_data(multires_cube, scale=True)

        multires_cube = multires_cube.chunk(chunking)

        for i in multires_cube.data_vars.keys():
            if i not in ['SCL', 'spatial_ref']:
                multires_cube[i] = multires_cube[i].astype("float32")
                del multires_cube[i].attrs['_FillValue']
                multires_cube[i].encoding = {"dtype": "uint16",
                                             "scale_factor": compute_scale_and_offset(multires_cube[i].values),
                                             "add_offset": 0.0,
                                             "_FillValue": 65535}
            elif i == 'SCL':
                # if 'SCL' in multires_cube.data_vars.keys():
                multires_cube[i] = multires_cube[i].astype("uint8")

        return multires_cube

    def harmonize_s2_data(self, data: xr.Dataset, scale: bool = True) -> xr.Dataset:
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

        bands = self.get_data_asset_keys()
        if "SCL" in bands:
            bands.remove("SCL")

        to_process = list(set(bands) & set(data.keys()))

        attrs = {p: data[p].attrs for p in to_process}

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

        for variable in no_change.keys():
            new[variable] = no_change[variable]

        for a in attrs.keys():
            if scale:
                attrs[a]['dtype'] = "float32"
                attrs[a]['_FillValue'] = np.nan
                attrs[a]['Harmonized'] = 'True'
            new[a].attrs = attrs[a]

        return new


class Modis13Q1Processor(STACItemProcessor):
    tilename = "modis:tile-id"
    datetime_id = "start_datetime"
    gridded = True

    indices = ["250m_16_days_NDVI",
               "250m_16_days_EVI"]

    reflectances = ["250m_16_days_MIR_reflectance",
                    "250m_16_days_NIR_reflectance",
                    "250m_16_days_red_reflectance",
                    "250m_16_days_blue_reflectance"]

    sinusoidal_pixel_spacing = 231.65635826

    indices_bands = {}
    for i in indices:
        indices_bands[i] = Band(
            name=i,
            data_type='int16',
            nodata_value=-3000,
            spatial_resolution=sinusoidal_pixel_spacing,
            continuous_measurement=True,
            # kwargs
            scale_factor=0.0001,
            valid_range=(-2000, 10000),
        )

    reflectance_bands = {}
    for i in reflectances:
        reflectance_bands[i] = Band(
            name=i,
            data_type='int16',
            nodata_value=-1000,
            spatial_resolution=sinusoidal_pixel_spacing,
            continuous_measurement=True,
            # kwargs
            scale_factor=0.0001,
            valid_range=(0, 10000),
        )

    other_bands = {
        "250m_16_days_VI_Quality" :
            Band(
                name="250m_16_days_VI_Quality",
                data_type='uint16',
                nodata_value=65535,
                spatial_resolution=sinusoidal_pixel_spacing,
                continuous_measurement=True,
                # kwargs
                valid_range=(0, 65534)
            ),
        "250m_16_days_pixel_reliability" :
            Band(
                name="250m_16_days_pixel_reliability",
                data_type='int16',
                nodata_value=-1,
                spatial_resolution=sinusoidal_pixel_spacing,
                continuous_measurement=False,
                # kwargs
                valid_range=(0, 3)
            ),
        "250m_16_days_relative_azimuth_angle" :
            Band(
                name="250m_16_days_relative_azimuth_angle",
                data_type='int16',
                nodata_value=-4000,
                spatial_resolution=sinusoidal_pixel_spacing,
                continuous_measurement=True,
                # kwargs
                scale_factor=0.01,
                valid_range=(-18000, 18000)
            ),
    }

    special_bands = {}
    all_bands = reflectance_bands | indices_bands | other_bands | special_bands
    supported_bands = list(all_bands.keys())

    def get_crs(self):
        return CRS.from_wkt(self.item.properties['proj:wkt2'])

    def get_bbox(self):
        return Polygon(self.item.properties['proj:geometry']['coordinates'][0])

    def snap_bbox_to_grid(self, bbox):
        # fits modis level 3 sinusoidal grid

        tile_size_m = 1111950.519667
        pixel_size = 231.656358263958
        n_tiles_h = 36
        n_tiles_v = 18
        x_min_global = -tile_size_m * (n_tiles_h / 2)
        y_min_global = -tile_size_m * (n_tiles_v / 2)

        xmin, ymin, xmax, ymax = bbox.bounds

        # Clip bbox to MODIS global bounds
        xmin = max(xmin, x_min_global)
        xmax = min(xmax, x_min_global + tile_size_m * n_tiles_h)
        ymin = max(ymin, y_min_global)
        ymax = min(ymax, y_min_global + tile_size_m * n_tiles_v)

        # Compute tile indices
        x_min_pix = (int(math.floor((xmin - x_min_global) / pixel_size) - n_tiles_h / 2 * 4800) * pixel_size)
        x_max_pix = (int(math.ceil((xmax - x_min_global) / pixel_size) - n_tiles_h / 2 * 4800) * pixel_size)
        y_min_pix = (int(math.floor((ymin - y_min_global) / pixel_size) - n_tiles_v / 2 * 4800) * pixel_size)
        y_max_pix = (int(math.ceil((ymax - y_min_global) / pixel_size) - n_tiles_v / 2 * 4800) * pixel_size)

        return box(x_min_pix, y_min_pix, x_max_pix, y_max_pix)


class ESAWorldCoverProcessor(STACItemProcessor):
    tilename = "esa_worldcover:product_tile"
    datetime_id = "start_datetime"
    x = 'longitude'
    y = 'latitude'
    gridded = True

    special_bands = {
        "map":
            Band(
                name="map",
                data_type="uint8",
                nodata_value=0,
                spatial_resolution=1 / 12000,
                continuous_measurement=False,
                long_name="ESA WorldCover product 2020/2021",
                flag_meanings=[
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
                flag_values=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
                metadata={
                    "color_bar_name": "LC Class",
                    "color_value_max": 100,
                    "color_value_min": 10,
                    "keywords": ["ESA WorldCover", "Classes"],
                },
            ),
        "input_quality":
            Band(
                name="input_quality",
                data_type="int16",
                nodata_value=-1,
                spatial_resolution=1 / 2000,
                continuous_measurement=False,
                long_name="Input Quality Map"
            )
    }

    supported_bands = list(special_bands.keys())
    all_bands = special_bands

    def snap_bbox_to_grid(self, bbox):
        pixel_size = 1 / 12000
        x_min_global = -180
        x_max_global = 180
        y_min_global = -90
        y_max_global = 82.75

        xmin, ymin, xmax, ymax = bbox.bounds

        # Clip bbox to MODIS global bounds
        xmin = max(xmin, x_min_global)
        xmax = min(xmax, x_max_global)
        ymin = max(ymin, y_min_global)
        ymax = min(ymax, y_max_global)

        # Compute tile indices
        x_min_pix = int(math.floor((xmin - x_min_global) / pixel_size)) * pixel_size - 180
        x_max_pix = int(math.ceil((xmax - x_min_global) / pixel_size)) * pixel_size - 180
        y_min_pix = int(math.floor((ymin - y_min_global) / pixel_size)) * pixel_size - 90
        y_max_pix = int(math.ceil((ymax - y_min_global) / pixel_size)) * pixel_size - 90

        return box(x_min_pix, y_min_pix, x_max_pix, y_max_pix)


class Sentinel3SynergyProcessor(STACItemProcessor):
    print('adapt the get_data_granules of sentinel_3 to include the folder, otherwise time steps overwrite each other!')
    datetime_id = "datetime"
    x = 'longitude'
    y = 'latitude'
    gridded = False
    cubing = 'custom'

    keys = [
        'geolocation'
        'syn-amin',
        'syn-flags',
        'syn-ato550',
        # 'tiepoints-olci',
        # 'tiepoints-meteo',
        # 'tiepoints-slstr-n',
        # 'tiepoints-slstr-o',
        'syn-angstrom-exp550',
        'syn-s1n-reflectance',
        'syn-s1o-reflectance',
        'syn-s2n-reflectance',
        'syn-s2o-reflectance',
        'syn-s3n-reflectance',
        'syn-s3o-reflectance',
        'syn-s5n-reflectance',
        'syn-s5o-reflectance',
        'syn-s6n-reflectance',
        'syn-s6o-reflectance',
        'syn-oa01-reflectance',
        'syn-oa02-reflectance',
        'syn-oa03-reflectance',
        'syn-oa04-reflectance',
        'syn-oa05-reflectance',
        'syn-oa06-reflectance',
        'syn-oa07-reflectance',
        'syn-oa08-reflectance',
        'syn-oa09-reflectance',
        'syn-oa10-reflectance',
        'syn-oa11-reflectance',
        'syn-oa12-reflectance',
        'syn-oa16-reflectance',
        'syn-oa17-reflectance',
        'syn-oa18-reflectance',
        'syn-oa21-reflectance',
        # 'syn-sdr-removed-pixels',
        # 'annotations-removed-pixels'
    ]

    def get_crs(self):
        return 4326

    @classmethod
    def get_supported_bands(cls):
        return cls.keys

    def get_data_coverage_geometry(self):
        return Polygon(*self.item.geometry["coordinates"])

    def snap_bbox_to_grid(self, bbox):
        # Sentinel-3 data is not in a grid format
        logging.warning("Sentinel-3 data is not in a grid format. No snapping applied.")
        return

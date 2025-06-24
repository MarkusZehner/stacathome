import logging
import os
from urllib import parse

import asf_search
import numpy as np
import pystac
import rasterio
import xarray as xr
from asf_search import Products
from asf_search.download.file_download_type import FileDownloadType
from pystac import Item
from pystac.utils import str_to_datetime
from rasterio.env import Env
from rio_stac.stac import bbox_to_geom, get_dataset_geom, get_projection_info, get_raster_info
from shapely import box, Polygon, transform

from stacathome.generic_utils import arange_bounds, create_utm_grid_bbox, get_transform
from .common import ASFResultProcessor, Band, STACItemProcessor


class OPERASentinel1RTCProcessor(ASFResultProcessor):
    dataset = "OPERA-S1"
    processingLevel = "RTC"
    platform = asf_search.PLATFORM.SENTINEL1
    gridded = True
    datetime_id = 'startTime'

    special_bands = {
        "mask": Band(
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
        "VV": Band(
            name="VV",
            data_type="float32",
            nodata_value=np.nan,
            spatial_resolution=30,
            continuous_measurement=True,
        ),
        "VH": Band(
            name="VH",
            data_type="float32",
            nodata_value=np.nan,
            spatial_resolution=30,
            continuous_measurement=True,
        ),
        "incidence_angle": Band(
            name="incidence_angle",
            data_type="float32",
            nodata_value=np.nan,
            spatial_resolution=30,
            continuous_measurement=True,
        ),
        "local_incidence_angle": Band(
            name="local_incidence_angle",
            data_type="float32",
            nodata_value=np.nan,
            spatial_resolution=30,
            continuous_measurement=True,
        ),
        "number_of_looks": Band(
            name="number_of_looks",
            data_type="float32",
            nodata_value=np.nan,
            spatial_resolution=30,
            continuous_measurement=True,
        ),
        "rtc_anf_gamma0_to_beta0": Band(
            name="rtc_anf_gamma0_to_beta0",
            data_type="float32",
            nodata_value=np.nan,
            spatial_resolution=30,
            continuous_measurement=True,
        ),
        "rtc_anf_gamma0_to_sigma0": Band(
            name="rtc_anf_gamma0_to_sigma0",
            data_type="float32",
            nodata_value=np.nan,
            spatial_resolution=30,
            continuous_measurement=True,
        ),
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
                        static_pre
                        + i.properties['operaBurstID'].replace('_', '-')
                        + static_post
                        + filler
                        + b
                        + file_type
                    )
                    urls.append(inc_url)

            dl_items[i] = urls

        dl_items_flat = [item for sublist in dl_items.values() for item in sublist]
        cls.provider.download_from_asf(dl_items_flat, path=path, processes=processes)

        for k in dl_items.keys():
            dl_items[k] = [os.path.join(path, os.path.split(parse.urlparse(paths).path)[1]) for paths in dl_items[k]]

        return dl_items

    @classmethod
    def generate_stac_items(cls, items):
        attrs_from_results = [
            'flightDirection',
            'pathNumber',
            'processingLevel',
            'url',
            'startTime',
            'platform',
            'orbit',
            'sensor',
            'subswath',
            'beamModeType',
            'operaBurstID',
        ]

        media_type = 'image/tiff'  # we could also use rio_stac.stac.get_media_type

        collection = "OPERAS1Product"

        return_items = []
        for i, paths in items.items():
            meta_from_item = {a: i.properties[a] for a in attrs_from_results}
            file_id = i.properties['file_id']

            assets = [
                {
                    "name": filename.split('/')[-1].replace('.tif', '').split('v1.0_')[1],
                    "path": filename,
                    "href": None,
                    "role": "data",
                }
                for filename in paths
                if filename.endswith('.tif')
            ]

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
                        dataset_geom = get_dataset_geom(
                            src_dst, densify_pts=0, precision=-1
                        )  # , geographic_crs=crs_proj)
                        bboxes.append(dataset_geom["bbox"])

                        proj_info = {
                            f"proj:{name}": value
                            for name, value in get_projection_info(src_dst).items()
                            if name in ['bbox', 'shape', 'transform']
                        }

                        raster_info = {"raster:bands": get_raster_info(src_dst, max_size=1024)}

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

            # item
            item = pystac.Item(
                id=file_id,
                geometry=bbox_to_geom(bbox),
                bbox=bbox,
                collection=collection,
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
            logging.info(
                'the current method does load static assets for each time step, for long time series just use VV, VH and mask bands!'
            )
        if len(items) > 100:
            logging.warning('large amount of assets, consider loading split in smaller time steps!')

        parameters = {
            "groupby": "solar_day",
            "fail_on_error": True,
        }

        if chunks:
            assert set(chunks.keys()) == {
                "time",
                self.x,
                self.y,
            }, f"Chunks must contain the dimensions 'time', {self.x}, {self.y}!"
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
        "vv": Band(
            name="vv",
            data_type="float32",
            nodata_value=-32768,
            spatial_resolution=10,
            continuous_measurement=True,
        ),
        "vh": Band(
            name="vh",
            data_type="float32",
            nodata_value=-32768,
            spatial_resolution=10,
            continuous_measurement=True,
        ),
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

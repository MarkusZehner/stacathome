import logging
from collections import defaultdict

import numpy as np
import xarray as xr
from shapely import box, Polygon, transform

from stacathome.generic_utils import arange_bounds, create_utm_grid_bbox, get_transform
from .common import Band, STACItemProcessor


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
        "atmos_opacity",
    ]

    special_bands = {
        'qa': Band(
            'qa',
            'int16',
            -9999,
            30,
            False,
            long_name="Surface Temperature Quality Assessment Band",
        ),
        'qa_pixel': Band(
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
                "Cirrus, high cloud",
            ],
            flag_values=[
                1,
                21824,
                21826,
                21888,
                21890,
                22080,
                22144,
                22280,
                23888,
                23952,
                24088,
                24216,
                24344,
                24472,
                30048,
                54596,
                54852,
                55052,
            ],
        ),
        'qa_aerosol': Band(
            'qa_aerosol',
            'uint8',
            1,
            30,
            False,
            long_name="",
        ),
        'qa_radsat': Band(
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
        created_box = create_utm_grid_bbox(bbox.bounds, grid_size, 15, 15)
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

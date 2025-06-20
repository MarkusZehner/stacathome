import os
import re
import logging
from urllib import parse
import numpy as np
import xarray as xr
from shapely import box
from earthaccess.results import DataGranule
import pystac
from pystac import Item
from pystac.utils import str_to_datetime
import rasterio
from rasterio.env import Env
from rio_stac.stac import (
    get_dataset_geom,
    get_projection_info,
    get_raster_info,
    bbox_to_geom,
)


from .common import Band, EarthAccessProcessor
from stacathome.generic_utils import create_utm_grid_bbox, arange_bounds, merge_item_datetime_by_timedelta, smallest_modulo_deviation


class ECOL2TLSTEProcessor(EarthAccessProcessor):
    collection = "ECO_L2T_LSTE.002"
    datetime_id = 'startTime'
    cubing = 'preferred'
    gridded = True
    overlap = True

    spatial_res = 70
    float32 = ['view_zenith', 'he_idht', 'LST', 'LST_err', 'EmisWB']
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

    def collect_covering_tiles_and_coverage(self, request_place, items):
        """
        to be called with the data granule
        use name info in granule to narrow down which data fits the request. 

        utm tile is in the name, but extent has to be drawn from the file metadata

        returns the filtered items sorted starting with the most covering tile
        """
        if not isinstance(self.item, dict):
            raise TypeError('expected item of type dict containing {DataGranule: [links]}!')

        items = self.filter_product_iteration_links(items)

        filtered_dict = {}
        for i in items:
            bbox = ECOL2TLSTEProcessor(i).get_bbox()
            mgrs_id = i['meta']['native-id'].split('_')[5]

            if mgrs_id not in filtered_dict:
                # maybe add overlap to remove not needed tiles even further
                filtered_dict[mgrs_id] = bbox.centroid.distance(request_place.centroid)

        min_dist_utm_code = min(filtered_dict)
        min_dist = filtered_dict[min_dist_utm_code]

        minx, miny, maxx, maxy = request_place.bounds

        # Compute edge lengths
        width = maxx - minx  # horizontal edge
        height = maxy - miny  # vertical edge

        # case 1: request tile is smaller than overlap: just get closest tile
        # case 2: if distance to centroid plus half of the request size (taking diagonal here) still within one tile
        # -> then we need just one tile
        # else easy workaround just return all
        # can be improved by using overlap, as this might in worst case download 4x the data
        if width < 0.090 and height < 0.090 or np.sqrt(width**2 + height**2) / 2 + min_dist < 1.:
            to_return = [i for i in items if i['meta']['native-id'].split('_')[5] == min_dist_utm_code]
        else:
            to_return = items

        return [to_return, ]

    @classmethod
    def download_tiles_to_file(cls, path, items, bands=None, processes=4):
        if isinstance(items, list):
            granules_links = {g: g.data_links() for g in items}
        else:
            granules_links = items

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
            file_id = i['meta']['native-id']
            file_id_split = file_id.split('_')
            meta_from_item["sensor"] = file_id_split[0][:3]
            meta_from_item["product_version"] = file_id_split[0][3:]
            meta_from_item["level_type"] = file_id_split[1]
            meta_from_item["geophysical_parameter"] = file_id_split[2]
            meta_from_item["orbit_number"] = file_id_split[3]
            meta_from_item["scene_id"] = file_id_split[4]
            meta_from_item["s2:mgrs_tile"] = file_id_split[5]
            meta_from_item["toa_str"] = file_id_split[6]
            meta_from_item["build_id"] = file_id_split[7]
            meta_from_item["product_iteration_nr"] = file_id_split[8]

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
                id=file_id,
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
        off_x = smallest_modulo_deviation(min(inherent_x), grid_size)
        off_y = smallest_modulo_deviation(min(inherent_y), grid_size)
        created_box = create_utm_grid_bbox(bbox.bounds, grid_size, off_x, off_y)
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
        return self.get_bbox()

    def load_cube(self, items, bands, geobox, split_by=100, chunks: dict = None):
        # raise NotImplementedError('Ecostress L2T.002 LSTE has observed shifts in x, therefore custom cubing to remedy!')
        if 'he_idht' in bands or 'view_zenith' in bands:
            logging.info('the current method does load static assets for each time step, for long time series consider not using he_idht or view_zenith bands!')
        if len(items) > 100:
            logging.warning('large amount of assets, consider loading split in smaller time steps!')

        items = self.sort_items_by_datetime(items)
        # items = merge_item_datetime_by_timedelta(items)

        parameters = {
            "groupby": merge_item_datetime_by_timedelta,
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
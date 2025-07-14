import datetime
import logging
from collections import defaultdict
from functools import cached_property

import numpy as np
import xarray as xr
import pystac
import shapely
from shapely import box, Polygon, transform
import odc.geo

from stacathome.generic_utils import (
    arange_bounds,
    compute_scale_and_offset,
    create_utm_grid_bbox,
    get_transform,
    merge_to_cover,
    most_common,
    resolve_best_containing,
    smallest_modulo_deviation,
)
from .base import BaseProcessor, register_default_processor
from stacathome.providers import BaseProvider
from stacathome.geo import wgs84_contains, wgs84_overlap_percentage, centroid_distance


class S2Item(pystac.Item):

    def __init__(self, item, validate: bool = True):
        self._item = item
        if validate:
            self._validate()

    def _validate(self):
        """
        Validate the item to ensure it has the necessary Sentinel-2 properties.
        """
        if self._item.geometry is None:
            raise ValueError('Item does not have a geometry.')

        if 's2:mgrs_tile' not in self._item.properties:
            raise ValueError("Item does not have 's2:mgrs_tile' property.")

        if 'proj:code' not in self._item.properties:
            raise ValueError("Item does not have 'proj:code' property.")


    def __getattr__(self, item):
        """
        Delegate attribute access to the underlying pystac.Item.
        """
        return getattr(self._item, item)

    @property
    def properties_s2(self):
        """
        Get the Sentinel-2 specific properties from the item.
        """
        return {k: v for k,v in self._item.properties.items() if k.startswith('s2:')}

    @property
    def proj_code(self):
        """
        Get the projection code from the item properties.
        """
        return self._item.properties['proj:code']

    @property
    def mgrs_tile(self):
        """
        Get the MGRS tile identifier from the item properties.
        """
        return self._item.properties['s2:mgrs_tile']

    @cached_property
    def geometry_shapely(self):
        """
        Get the geometry of the item as a shapely object.
        """
        return shapely.geometry.shape(self._item.geometry)

    @cached_property
    def bbox_shapely(self):
        """
        Get the bounding box of the item as a shapely object.
        """
        return box(*self._item.bbox)

    @cached_property
    def geometry_odc_geometry(self):
        """
        Get the geometry of the item as a odc.geo Geometry object.
        """
        return odc.geo.Geometry(shapely.geometry.shape(self._item.geometry), '4326')

    @cached_property
    def bbox_odc_geometry(self):
        """
        Get the bounding box of the item as a odc.geo Geometry object.
        """
        return odc.geo.Geometry(box(*self._item.bbox), '4326')


def s2_pc_filter_newest_processing_time(items: list) -> list:
    """
    Returns the newest veriosn of S2 L2A items using the processing time from the ID.
    The ID is expected to be in the format 'S2A_MSIL2A_20220101T000000_N0509_R123_T123456_20220101T000000'.
    """
    filtered = {}
    for item in items:
        native_id = item._item.id
        base_name, process_time = native_id.rsplit('_', 1)
        if base_name not in filtered or process_time > filtered[base_name][0]:
            filtered[base_name] = (process_time, item)
    return [v[1] for v in filtered.values()]

def s2_pc_filter_coverage(items:list , roi: odc.geo.Geometry) -> list:
    """
    Filter Sentinel-2 items based on their coverage of the area of interest.
    Returns a list items required to cover the area of interest.
    """
    mgrs_tiles = defaultdict(list)
    for item in items:
        mgrs_tiles[item.mgrs_tile].append(item)

    centroid_distances = {}
    latitude_distance_from_utm_center = 500000  # not a good candiate if > half of a utm zone
    return_items = None
    
    for v in mgrs_tiles.values():
        bbox = v[0].bbox_odc_geometry
        proj = v[0].proj_code
        centroid_latitude_distance_from_utm_center = abs(bbox.to_crs(proj).centroid.points[0][0] - 500000)

        if wgs84_contains(bbox, roi, proj) and \
            latitude_distance_from_utm_center > centroid_latitude_distance_from_utm_center:
            latitude_distance_from_utm_center = centroid_latitude_distance_from_utm_center
            return_items = v

        centroid_distances[v[0].mgrs_tile] = centroid_distance(bbox, roi)

    if return_items is None:
        centroid_distances = sorted(centroid_distances.items(), key=lambda x: x[1])
        iterative_shape = None
        return_items = []
        for k, _ in centroid_distances:
            if not iterative_shape:
                print(mgrs_tiles[k])
                iterative_shape = mgrs_tiles[k][0].bbox_odc_geometry
            else:
                iterative_shape = iterative_shape.union(mgrs_tiles[k][0].bbox_odc_geometry)

            return_items.extend(mgrs_tiles[k])

            if wgs84_contains(iterative_shape, roi, proj):
                return return_items
    return return_items

class Sentinel2L2AProcessor(BaseProcessor):
     
    def __init__(self, convert_to_f32: bool = False, adjust_baseline: bool = True):
        """
        Sentinel-2 L2A processor for downloading and processing Sentinel-2 Level 2A data.
        """
        self.convert_to_f32 = convert_to_f32
        self.adjust_baseline = adjust_baseline
    
    def filter_items(self, provider: BaseProvider, roi: odc.geo.Geometry, items: pystac.ItemCollection) -> pystac.ItemCollection:
        """

        Filter Sentinel-2 items based on the area of interest and the newest processing time.
        This function filters the items to ensure they cover the area of interest and selects the newest processing time for each item.
        Reoders the Items by the tile id, first item(s) corresponding to the tile closest to the roi.

        """


        s2_items = [S2Item(item) for item in items]

        s2_items = s2_pc_filter_newest_processing_time(s2_items)

        s2_items = s2_pc_filter_coverage(s2_items, roi)

        return pystac.ItemCollection(
            items=s2_items,
            clone_items=False,
            extra_fields=items.extra_fields,
        )

    def load_items(
        self, provider: BaseProvider, 
        roi: odc.geo.geom.Geometry, 
        items: pystac.ItemCollection, 
        variables: list[str] | None = None,
    ) -> xr.Dataset:
        # check if variables are provided, if not use all supported bands -> provider based
        # create geobox from fitting item (otherwise will use first item) -> filter gives back matching ones first
        
        
        
        
        pass


    # def load_items(self, provider: BaseProvider, area_of_interest: shapely.Geometry, items: pystac.ItemCollection) -> xr.Dataset:
    #     # get one item with the same tile_id for the geobox
    #     item = next(i for i in collect_coverage_from if self.__class__(i).get_tilename_value() == tile_ids[0])
    #     geobox = self.__class__(item).get_geobox(request_place)
        
    #     parameters = {
    #         "groupby": "solar_day",
    #         "fail_on_error": False,
    #     }

    #     if chunks:
    #         assert set(chunks.keys()) == {"time", "x", "y"}, "Chunks must contain the dimensions 'time', 'x', 'y'!"
    #         parameters['chunks'] = chunks

    #     multires_cube = {}
    #     for gb, band_subset in geobox.items():
    #         req_bands = set(band_subset) & set(bands)
    #         if len(req_bands) == 0:
    #             continue
    #         resampling = self.get_resampling_per_band(gb.resolution.x)
    #         dtypes = self.get_dtype_per_band(gb.resolution.x)
    #         resampling = {k: resampling[k] for k in req_bands if k in resampling}
    #         dtypes = {k: dtypes[k] for k in req_bands if k in dtypes}

    #         parameters['bands'] = req_bands
    #         parameters['geobox'] = gb
    #         parameters['resampling'] = resampling
    #         parameters['dtype'] = dtypes
    #         if split_by is not None and len(items) > split_by:
    #             split_items = self.split_items_keep_solar_days_together(items, split_by)
    #             cube = []
    #             for split in split_items:
    #                 parameters['items'] = split
    #                 cube.append(self.provider.download_cube(parameters))
    #             cube = xr.concat(cube, dim="time")
    #         else:
    #             parameters['items'] = items
    #             cube = self.provider.download_cube(parameters)
    #         attrs = self.get_band_attributes(req_bands)
    #         for band in cube.keys():
    #             cube[band].attrs = attrs[band]

    #         multires_cube[int(gb.resolution.x)] = cube

    #     # remove empty images, could be moved into separate function
    #     # get coarsest resolution first band and check where mean value is 0:
    #     coarsest = max(multires_cube.keys())
    #     first_var = list(multires_cube[coarsest].data_vars.keys())[0]
    #     mean_over_time = multires_cube[coarsest][first_var].mean(dim=['x', 'y'])
    #     na_value = multires_cube[coarsest][first_var].attrs['_FillValue']
    #     mask_over_time = np.where(mean_over_time != na_value)[0]

    #     chunking = {"time": 2 if not chunks else chunks['time']}
    #     for spat_res in multires_cube.keys():
    #         # remove empty images, could be moved into separate function
    #         multires_cube[spat_res] = multires_cube[spat_res].isel(time=mask_over_time)

    #         chunking[f'x{spat_res}'] = -1 if not chunks else chunks['x']
    #         chunking[f'y{spat_res}'] = -1 if not chunks else chunks['y']
    #         multires_cube[spat_res] = multires_cube[spat_res].rename({'x': f'x{spat_res}', 'y': f'y{spat_res}'})

    #     multires_cube = xr.merge(multires_cube.values())



    #     return multires_cube

        
    # def postprocess_data(self, provider: BaseProvider, area_of_interest: shapely.Geometry, data: xr.Dataset) -> xr.Dataset:
    #     # mask by SCL?
    #     # remove values time steps with no data per SCL layer?

    #     # this changes int16 to float32 could also be seen as different function
    #     multires_cube = self.harmonize_s2_data(multires_cube, scale=True)

    #     multires_cube = multires_cube.chunk(chunking)

    #     for i in multires_cube.data_vars.keys():
    #         if i not in ['SCL', 'spatial_ref']:
    #             multires_cube[i] = multires_cube[i].astype("float32")
    #             del multires_cube[i].attrs['_FillValue']
    #             multires_cube[i].encoding = {
    #                 "dtype": "uint16",
    #                 "scale_factor": compute_scale_and_offset(multires_cube[i].values),
    #                 "add_offset": 0.0,
    #                 "_FillValue": 65535,
    #             }
    #         elif i == 'SCL':
    #             # if 'SCL' in multires_cube.data_vars.keys():
    #             multires_cube[i] = multires_cube[i].astype("uint8")


    # tilename = "s2:mgrs_tile"
    # datetime_id = "datetime"
    # gridded = True
    # overlap = True

    # supported_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"]
    # special_bands = {
    #     'SCL': Band(
    #         'SCL',
    #         'uint8',
    #         0,
    #         20,
    #         False,
    #         long_name="Scene Classification Layer",
    #         flag_meanings=[
    #             "Saturated / Defective",
    #             "Dark Area Pixels",
    #             "Cloud Shadows",
    #             "Vegetation",
    #             "Bare Soils",
    #             "Water",
    #             "Clouds low probability / Unclassified",
    #             "Clouds medium probability",
    #             "Clouds high probability",
    #             "Cirrus",
    #             "Snow / Ice",
    #         ],
    #         flag_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #     )
    # }

    # def get_assets_as_bands(self):
    #     present_bands = self.get_data_asset_keys()
    #     present_supported_bands = sorted(set(self.supported_bands) & set(present_bands))

    #     bands = []
    #     for b in present_supported_bands:
    #         special = self.special_bands.get(b, None)
    #         if special:
    #             bands.append(special)
    #         else:
    #             info = self.item.assets[b].to_dict()
    #             eo_bands = info.get('eo:bands')
    #             if isinstance(eo_bands, list) and len(eo_bands) == 1:
    #                 eo_bands = eo_bands[0]
    #             bands.append(
    #                 Band(
    #                     b,
    #                     "uint16",
    #                     0,
    #                     info.get('gsd', 0),
    #                     True,
    #                     long_name=eo_bands.get('description'),
    #                     center_wavelenght=eo_bands.get('center_wavelength'),
    #                     full_width_half_max=eo_bands.get('full_width_half_max'),
    #                 )
    #             )

    #     return bands

    # def get_bbox(self):
    #     key = self.get_data_asset_keys()[0]
    #     return box(*self.item.assets[key].extra_fields['proj:bbox'])

    # def get_band_attributes(self, bands: list[str] | set[str]):
    #     info = self.get_assets_as_bands()
    #     bandattributes = {}
    #     for i in info:
    #         d = i.to_dict()
    #         bandattributes[d['Name']] = d
    #     bandattributes = {b: bandattributes[b] for b in bands if b in bandattributes}
    #     return bandattributes

    # def get_data_asset_keys(self, role="data"):
    #     key = list(self.item.get_assets(role=role).keys())
    #     assert len(key) > 0, "No data assets found!"
    #     return key

    # def get_data_coverage_geometry(self):
    #     return transform(Polygon(*self.item.geometry["coordinates"]), get_transform(4326, self.get_crs()))

    # def collect_covering_tiles_and_coverage(self, request_place, item_limit=12, items=None, request_time=None):
    #     # move this into the processor classes?
    #     tile_ids_per_collection = {}

    #     if items is None and all([request_time is not None, request_place is not None]):
    #         items = self.provider.request_items(
    #             self.item.collection_id, request_time=request_time, request_place=request_place, max_items=item_limit
    #         )

    #         if items is None:
    #             raise ValueError("No items found for the given request parameters.")
    #     else:
    #         filter = True

    #     if len(items) < item_limit:
    #         logging.warning(
    #             f"Less than {item_limit} items found for {self.item.collection_id} in {request_place} "
    #             f"and {request_time}"
    #         )

    #     collect_coverage_from = items[: min(len(items), item_limit)]

    #     by_tile = defaultdict(list)

    #     for i in collect_coverage_from:
    #         item = self.__class__(i)
    #         by_tile[item.get_tilename_value()].append(
    #             [
    #                 item.get_crs(),
    #                 item.contains_shape(request_place),
    #                 item.centroid_distance_to(request_place),
    #                 item.overlap_percentage(request_place),
    #                 item.get_bbox(),
    #             ]
    #         )

    #     # Reduce each group using majority voting
    #     by_tile_filtered = [[tile_id] + [most_common(attr) for attr in zip(*vals)] for tile_id, vals in by_tile.items()]

    #     # First, try finding a containing item
    #     best = resolve_best_containing(by_tile_filtered)
    #     if best:
    #         found_tiles = [best]
    #     else:
    #         found_tiles = merge_to_cover(by_tile_filtered, request_place)

    #     tile_ids = [t[0] for t in found_tiles]

    #     # get one item with the same tile_id for the geobox
    #     item = next(i for i in collect_coverage_from if self.__class__(i).get_tilename_value() == tile_ids[0])

    #     geobox = self.__class__(item).get_geobox(request_place)

    #     if filter:
    #         filtered_items = [i for i in items if self.__class__(i).get_tilename_value() in tile_ids]

    #     tile_ids_per_collection = {'tile_id': tile_ids, 'geobox': geobox}

    #     if filter:
    #         return [filtered_items, tile_ids_per_collection]
    #     else:
    #         return tile_ids_per_collection

    # def snap_bbox_to_grid(self, bbox, grid_size=60):
    #     inherent_bbox = self.get_bbox()
    #     inherent_x, inherent_y = arange_bounds(inherent_bbox.bounds, grid_size)
    #     off_x = smallest_modulo_deviation(min(inherent_x), grid_size)
    #     off_y = smallest_modulo_deviation(min(inherent_y), grid_size)
    #     created_box = create_utm_grid_bbox(bbox.bounds, grid_size, off_x, off_y)
    #     created_x, created_y = arange_bounds(created_box.bounds, grid_size)

    #     # Determine the overlapping grid coordinates
    #     shared_x = set(created_x) & set(inherent_x)
    #     shared_y = set(created_y) & set(inherent_y)

    #     if len(created_x) >= len(inherent_x):
    #         overlap_x = created_x[(created_x >= min(inherent_x)) & (created_x <= max(inherent_x))]
    #         overlap_y = created_y[(created_y >= min(inherent_y)) & (created_y <= max(inherent_y))]

    #     else:
    #         overlap_x = inherent_x[(inherent_x >= min(created_x)) & (inherent_x <= max(created_x))]
    #         overlap_y = inherent_y[(inherent_y >= min(created_y)) & (inherent_y <= max(created_y))]

    #     if len(shared_x) != len(overlap_x) or len(shared_y) != len(overlap_y):
    #         raise ValueError("Grid of created box does not fully align with inherent grid")

    #     return created_box

    # def load_cube(self, items, bands, geobox, split_by=100, chunks: dict = None):
    #     # most of this until the s2 specific harmonization could be moved to the base class

    #     # implement this in multiprocess?

    #     if len(items) > 100:
    #         logging.warning('large amount of assets, consider loading split in smaller time steps!')

    #     parameters = {
    #         "groupby": "solar_day",
    #         "fail_on_error": False,
    #     }

    #     if chunks:
    #         assert set(chunks.keys()) == {"time", "x", "y"}, "Chunks must contain the dimensions 'time', 'x', 'y'!"
    #         parameters['chunks'] = chunks

    #     multires_cube = {}
    #     for gb, band_subset in geobox.items():
    #         req_bands = set(band_subset) & set(bands)
    #         if len(req_bands) == 0:
    #             continue
    #         resampling = self.get_resampling_per_band(gb.resolution.x)
    #         dtypes = self.get_dtype_per_band(gb.resolution.x)
    #         resampling = {k: resampling[k] for k in req_bands if k in resampling}
    #         dtypes = {k: dtypes[k] for k in req_bands if k in dtypes}

    #         parameters['bands'] = req_bands
    #         parameters['geobox'] = gb
    #         parameters['resampling'] = resampling
    #         parameters['dtype'] = dtypes
    #         if split_by is not None and len(items) > split_by:
    #             split_items = self.split_items_keep_solar_days_together(items, split_by)
    #             cube = []
    #             for split in split_items:
    #                 parameters['items'] = split
    #                 cube.append(self.provider.download_cube(parameters))
    #             cube = xr.concat(cube, dim="time")

    #         else:
    #             parameters['items'] = items
    #             cube = self.provider.download_cube(parameters)
    #         attrs = self.get_band_attributes(req_bands)
    #         for band in cube.keys():
    #             cube[band].attrs = attrs[band]

    #         multires_cube[int(gb.resolution.x)] = cube

    #     # remove empty images, could be moved into separate function
    #     # get coarsest resolution first band and check where mean value is 0:
    #     coarsest = max(multires_cube.keys())
    #     first_var = list(multires_cube[coarsest].data_vars.keys())[0]
    #     mean_over_time = multires_cube[coarsest][first_var].mean(dim=['x', 'y'])
    #     na_value = multires_cube[coarsest][first_var].attrs['_FillValue']
    #     mask_over_time = np.where(mean_over_time != na_value)[0]

    #     chunking = {"time": 2 if not chunks else chunks['time']}
    #     for spat_res in multires_cube.keys():
    #         # remove empty images, could be moved into separate function
    #         multires_cube[spat_res] = multires_cube[spat_res].isel(time=mask_over_time)

    #         chunking[f'x{spat_res}'] = -1 if not chunks else chunks['x']
    #         chunking[f'y{spat_res}'] = -1 if not chunks else chunks['y']
    #         multires_cube[spat_res] = multires_cube[spat_res].rename({'x': f'x{spat_res}', 'y': f'y{spat_res}'})

    #     multires_cube = xr.merge(multires_cube.values())

    #     # mask by SCL?
    #     # remove values time steps with no data per SCL layer?

    #     # this changes int16 to float32 could also be seen as different function
    #     multires_cube = self.harmonize_s2_data(multires_cube, scale=True)

    #     multires_cube = multires_cube.chunk(chunking)

    #     for i in multires_cube.data_vars.keys():
    #         if i not in ['SCL', 'spatial_ref']:
    #             multires_cube[i] = multires_cube[i].astype("float32")
    #             del multires_cube[i].attrs['_FillValue']
    #             multires_cube[i].encoding = {
    #                 "dtype": "uint16",
    #                 "scale_factor": compute_scale_and_offset(multires_cube[i].values),
    #                 "add_offset": 0.0,
    #                 "_FillValue": 65535,
    #             }
    #         elif i == 'SCL':
    #             # if 'SCL' in multires_cube.data_vars.keys():
    #             multires_cube[i] = multires_cube[i].astype("uint8")

    #     return multires_cube

    # def harmonize_s2_data(self, data: xr.Dataset, scale: bool = True) -> xr.Dataset:
    #     """
    #     Harmonize new Sentinel-2 data to the old baseline. Data after 25-01-2022 is clipped
    #     to 1000 and then subtracted by 1000.
    #     From https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change
    #     adjusted to odc-stac, using different variables for each band.

    #     Parameters
    #     ----------
    #     data: xarray.DataArray
    #         A DataArray with four dimensions: time, band, y, x

    #     Returns
    #     -------
    #     harmonized: xarray.DataArray
    #         A DataArray with all values harmonized to the old
    #         processing baseline.
    #     """
    #     cutoff = datetime.datetime(2022, 1, 25)
    #     offset = 1000

    #     bands = self.get_data_asset_keys()
    #     if "SCL" in bands:
    #         bands.remove("SCL")

    #     to_process = list(set(bands) & set(data.keys()))

    #     attrs = {p: data[p].attrs for p in to_process}

    #     no_change = data.drop_vars(to_process)
    #     old = data[to_process].sel(time=slice(cutoff))
    #     new_harmonized = data[to_process].sel(time=slice(cutoff, None)).clip(offset)
    #     new_harmonized -= offset

    #     new = xr.concat([old, new_harmonized], "time")
    #     if scale:
    #         new = new.where(new != 0)
    #         new = (new * 0.0001).astype("float32")
    #     else:
    #         new = new.astype("uint16")

    #     for variable in no_change.keys():
    #         new[variable] = no_change[variable]

    #     for a in attrs.keys():
    #         if scale:
    #             attrs[a]['dtype'] = "float32"
    #             attrs[a]['_FillValue'] = np.nan
    #             attrs[a]['Harmonized'] = 'True'
    #         new[a].attrs = attrs[a]

    #     return new

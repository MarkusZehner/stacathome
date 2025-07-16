import datetime
from collections import defaultdict
from functools import cached_property

import numpy as np
import xarray as xr
import pystac
import shapely
from shapely import box
import odc.geo.geom as geom
from xarray import Dataset

from stacathome.providers import BaseProvider
import stacathome.geo as geo
from .base import SimpleProcessor


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
        return geom.Geometry(shapely.geometry.shape(self._item.geometry), '4326')

    @cached_property
    def bbox_odc_geometry(self):
        """
        Get the bounding box of the item as a odc.geo Geometry object.
        """
        return geom.Geometry(box(*self._item.bbox), '4326')


def s2_pc_filter_newest_processing_time(items: list) -> list:
    """
    Returns the newest veriosn of S2 L2A items using the processing time from the ID.
    The ID is expected to be in the format 'S2A_MSIL2A_20220101T000000_N0509_R123_T123456_20220101T000000'.
    """
    filtered = {}
    for item in items:
        native_id = item.id
        base_name, process_time = native_id.rsplit('_', 1)
        if base_name not in filtered or process_time > filtered[base_name][0]:
            filtered[base_name] = (process_time, item)
    return [v[1] for v in filtered.values()]

def s2_pc_filter_coverage(items:list , roi: geom.Geometry) -> list:
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

        if geo.wgs84_contains(bbox, roi) and \
            latitude_distance_from_utm_center > centroid_latitude_distance_from_utm_center:
            latitude_distance_from_utm_center = centroid_latitude_distance_from_utm_center
            return_items = v

        centroid_distances[v[0].mgrs_tile] = geo.wgs84_centroid_distance(bbox, roi)

    if return_items is None:
        centroid_distances = sorted(centroid_distances.items(), key=lambda x: x[1])
        iterative_shape = None
        return_items = []
        for k, _ in centroid_distances:
            if not iterative_shape:
                iterative_shape = mgrs_tiles[k][0].bbox_odc_geometry
            else:
                iterative_shape = iterative_shape.union(mgrs_tiles[k][0].bbox_odc_geometry)

            return_items.extend(mgrs_tiles[k])

            if geo.wgs84_contains(iterative_shape, roi):
                return return_items
    return return_items


def s2_pc_filter_geometry_coverage(items:list , roi: geom.Geometry) -> list:
    return_list = []
    for item in items:
        i = S2Item(item) if not isinstance(item, S2Item) else item
        if geo.wgs84_intersects(i.geometry_odc_geometry, roi, i.proj_code):
            return_list.append(item)
    return return_list


class Sentinel2L2AProcessor(SimpleProcessor):

    def __init__(self, convert_to_f32: bool = False, adjust_baseline: bool = True):
        """
        Sentinel-2 L2A processor for downloading and processing Sentinel-2 Level 2A data.
        """
        self.convert_to_f32 = convert_to_f32
        self.adjust_baseline = adjust_baseline
    
    def filter_items(self, provider: BaseProvider, roi: geom.Geometry, items: pystac.ItemCollection) -> pystac.ItemCollection:
        """

        Filter Sentinel-2 items based on the area of interest and the newest processing time.
        This function filters the items to ensure they cover the area of interest and selects the newest processing time for each item.
        Reoders the Items by the tile id, first item(s) corresponding to the tile closest to the roi.

        """
        s2_items = [S2Item(item) for item in items]
        s2_items = s2_pc_filter_newest_processing_time(s2_items)
        s2_items = s2_pc_filter_coverage(s2_items, roi)
        s2_items = s2_pc_filter_geometry_coverage(s2_items, roi)
        return pystac.ItemCollection(
            items=s2_items,
            clone_items=False,
            extra_fields=items.extra_fields,
        )
    
    def postprocess_data(self, provider, roi, data:Dataset, harmonize = False,
                         mask_by_scl: bool = False, valid_scl_values:list[int]| None=None) -> Dataset:
        data = filter_no_data_timesteps(data)
        data = rename_s2_coords(data)
        if harmonize:
            data = harmonize_s2_data(data)
            for variable in data.data_vars:
                data[variable].attrs['harmonized'] = True
                
        # if mask_by_scl:
        #     data = mask_data_by_scl(data, valid_scl_values)
            
        return data

def mask_data_by_scl(data:Dataset, valid_scl_values:list[int] | None = None):
    raise NotImplementedError
    if not 'SCL' in data.data_vars:
        raise ValueError('"SCL" variable not in data_vars, which is required by mask_by_scl')
    valid_scl_values = valid_scl_values if valid_scl_values else [4, 5, 6, 7, 11]

    scl_mask = xr.where(data.SCL.isin(valid_scl_values), 1, 0)
    coords_res = _get_coord_name_and_resolution(data)

    for res, coord_names in coords_res.items():
        # need to resample to the present data variables to mask
        pass

        data = xr.where(scl_mask == 1, data, 0)
    return None


def filter_no_data_timesteps(data:Dataset, indicator_variable:str | None = None):
    if not indicator_variable:
        coords_res = _get_coord_name_and_resolution(data)
        coarsest_axes = coords_res[max(coords_res)]
        for data_var in data.data_vars.values():
            if coarsest_axes[0] in data_var.dims:
                indicator_variable = data_var.name
                break

    mean_over_time = data[indicator_variable].mean(dim=coarsest_axes)
    na_value = data[indicator_variable].attrs['nodata_value']
    mask_over_time = np.where(mean_over_time != na_value)[0]
    return data.isel(time=mask_over_time)


def rename_s2_coords(data:Dataset):
    coord_names_resolution = _get_coord_name_and_resolution(data)
    rename_dict = {}
    for resolution, coord_names in coord_names_resolution.items():
        renamed_coords = {
            coord: f"{coord.split('_')[0]}_{str(int(resolution))}"
            for coord in coord_names
        }
        rename_dict |= renamed_coords
    return data.rename(rename_dict)
      
      
def _get_coord_name_and_resolution(data:Dataset) -> dict[float, str]:
    coord_names_resolution = defaultdict(list)
    for coord in data.coords.values():
        if coord.name.startswith('x') or coord.name.startswith('y'):
            coord_names_resolution[abs(coord.attrs['resolution'])].append(coord.name)
    return dict(coord_names_resolution)


def harmonize_s2_data(data: Dataset, scale: bool = False) -> xr.Dataset:
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

    bands = list(data.data_vars)
    do_not_harmonize = ['SCL', 'AOT', 'WVP']
    for var in do_not_harmonize:
        if var in bands:
            bands.remove(var)

    prev_dtype = str(data[bands[0]].dtype)
    
    to_process = list(set(bands) & set(data.keys()))

    attrs = {p: data[p].attrs for p in to_process}

    no_change = data.drop_vars(to_process)
    old = data[to_process].sel(time=slice(cutoff))
    new_harmonized = data[to_process].sel(time=slice(cutoff, None)).clip(offset)
    new_harmonized -= offset

    new = xr.concat([old, new_harmonized], "time")
    if scale:
        new = new.where(new != 0)
        new = new.astype("float32")
        new *= 0.0001
    else:
        new = new.astype(prev_dtype)

    for variable in no_change.keys():
        new[variable] = no_change[variable]
    
    for k, v in attrs.items():
        new[k].attrs = v

    return new

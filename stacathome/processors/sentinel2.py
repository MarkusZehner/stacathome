import datetime

import odc.geo.geom as geom
import pystac
import xarray as xr

from stacathome.providers import SimpleProvider
from .base import register_default_processor, SimpleProcessor
from .common import (
    _get_coord_name_and_resolution,
    filter_no_data_timesteps,
    mgrs_tiled_overlap_filter_coverage,
    MGRSTiledItem,
)


def s2_pc_filter_newest_processing_time(items: list[MGRSTiledItem]) -> list[MGRSTiledItem]:
    """
    Returns the newest version of S2 L2A items using the processing time from the ID.
    The ID is expected to be in the format
    'S2A_MSIL2A_20220101T000000_R123_T123456_20220101T000000' (note 'NXXXX' is missing)
    or 'S2A_MSIL2A_20220101T000000_N0509_R123_T123456_20220101T000000'.
    """
    filtered = {}
    for item in items:
        native_id = item.id.split('_')
        if len(native_id) == 6:
            # planetary laundered the Processing Baseline number from its ids
            mission_id, prod_lvl, datatake_sensing_start_time, rel_orbit, mgrs_tile, process_time = native_id
        elif len(native_id) == 7:
            # CDSE has the Processing Baseline number
            mission_id, prod_lvl, datatake_sensing_start_time, _, rel_orbit, mgrs_tile, process_time = native_id
        else:
            raise ValueError(f'Id if MGRSTiledItem does not match known providers with {item.id}')
        identifier = '_'.join([mission_id, prod_lvl, datatake_sensing_start_time, rel_orbit, mgrs_tile])
        if identifier not in filtered or process_time > filtered[identifier][0]:
            filtered[identifier] = (process_time, item)
    return [v[1] for v in filtered.values()]


class Sentinel2L2AProcessor(SimpleProcessor):

    def __init__(self, convert_to_f32: bool = False, adjust_baseline: bool = True):
        """
        Sentinel-2 L2A processor for downloading and processing Sentinel-2 Level 2A data.
        """
        self.convert_to_f32 = convert_to_f32
        self.adjust_baseline = adjust_baseline

    def filter_items(
        self,
        provider: SimpleProvider,
        roi: geom.Geometry,
        items: pystac.ItemCollection,
        variables: list[str] | None = None,
        temp_path: str | None = None,
    ) -> pystac.ItemCollection:
        """

        Filter Sentinel-2 items based on the area of interest and the newest processing time.
        This function filters the items to ensure they cover the area of interest and selects the newest processing time for each item.
        Reoders the Items by the tile id, first item(s) corresponding to the tile closest to the roi.

        """
        s2_items = [MGRSTiledItem(item) for item in items]
        s2_items = s2_pc_filter_newest_processing_time(s2_items)
        s2_items = mgrs_tiled_overlap_filter_coverage(s2_items, roi)
        return pystac.ItemCollection(
            items=s2_items,
            clone_items=False,
            extra_fields=items.extra_fields,
        )

    def postprocess_data(
        self,
        provider,
        roi,
        data: xr.Dataset,
    ) -> xr.Dataset:
        data = filter_no_data_timesteps(data)
        if self.adjust_baseline:
            data = harmonize_s2_data(data)
        return data


def mask_data_by_scl(data: xr.Dataset, valid_scl_values: list[int] | None = None) -> xr.Dataset:
    if 'SCL' not in data.data_vars:
        raise ValueError('"SCL" variable not in data_vars, which is required by mask_by_scl')
    valid_scl_values = valid_scl_values if valid_scl_values else [4, 5, 6, 7, 11]

    scl_mask = xr.where(data.SCL.isin(valid_scl_values), 1, 0)

    coords_res = _get_coord_name_and_resolution(data)

    scl_dims = data["SCL"].dims
    x_scl = [d for d in scl_dims if d.startswith('x')]
    y_scl = [d for d in scl_dims if d.startswith('y')]

    if not len(x_scl) == len(y_scl) == 1:
        raise ValueError('Ambiguous x and y coords in SCL band!')
    x_scl = x_scl[0]
    y_scl = y_scl[0]

    for res, coord_names in coords_res.items():
        x_name = [d for d in coord_names if d.startswith('x')]
        y_name = [d for d in coord_names if d.startswith('y')]
        if not len(x_name) == len(y_name) == 1:
            raise ValueError('Ambiguous x and y coords in SCL band!')
        x_name = x_name[0]
        y_name = y_name[0]

        res_vars = [var for var in data.data_vars if set(coord_names).issubset(data[var].dims)]

        if not res_vars:
            continue

        if res is not abs(data[res_vars[0]][x_name].attrs['resolution']):
            scl_mask_res = scl_mask.interp(
                {x_scl: data[res_vars[0]][x_name], y_scl: data[res_vars[0]][y_name]}, method='nearest'
            )
        else:
            scl_mask_res = scl_mask

        for var in res_vars:
            if var == 'SCL':
                continue
            data[var] = data[var].where(scl_mask_res == 1)

    return data


def harmonize_s2_data(data: xr.Dataset, scale: bool = False) -> xr.Dataset:
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
    do_not_harmonize = ['SCL', 'AOT', 'WVP', 'TCI']

    bands = [b for b in bands if not any(var in b for var in do_not_harmonize)]

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


register_default_processor('planetary_computer', 'sentinel-2-l2a', Sentinel2L2AProcessor)
register_default_processor('cdse', 'sentinel-2-l2a', Sentinel2L2AProcessor)

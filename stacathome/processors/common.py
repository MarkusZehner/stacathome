from collections import defaultdict, namedtuple
from functools import cached_property

import numpy as np
import odc.geo.geom as geom
import pystac
import shapely
import xarray as xr
from shapely import box

import stacathome.geo as geo


class MGRSTiledItem(pystac.Item):

    def __init__(self, item, validate: bool = True):
        self._item = item
        if validate:
            self._validate()

    def _validate(self):
        if self._item.geometry is None:
            raise ValueError('Item does not have a geometry.')
        if not self.mgrs_tile:
            raise ValueError("Item does not contain a tile-id property.")
        if not get_property(self._item, 'proj:code'):
            raise ValueError("Item does not have 'proj:code' property.")

    def __getattr__(self, item):
        """
        Delegate attribute access to the underlying pystac.Item.
        """
        return getattr(self._item, item)

    @property
    def proj_code(self):
        """
        Get the projection code from the item properties.
        """
        return get_property(self._item, 'proj:code')

    @property
    def mgrs_tile(self):
        """
        Get the MGRS tile identifier from the item properties.
        """
        mgrs_tile = get_property(self._item, 's2:mgrs_tile')
        if not mgrs_tile:
            mgrs_tile = get_property(self._item, 'grid:code').split('-')[-1]
        return mgrs_tile

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


def get_property(item: pystac.Item, property_name: str, asset_name: str = None) -> str | None:
    """
    Get a defined property from a STAC item or its assets.

    args:
        item: The STAC item to get the property from.
        property_name: The name of the property to get.
        asset_name: The name of the asset to get the property from. If None, the
                    property is searched in all assets and the item itself.
    returns:
        The value of the property, or None if not found.
    """
    if asset_name:
        asset = item.assets.get(asset_name)
        if asset is None:
            raise ValueError(f"Asset '{asset_name}' not found in item.")
        item_property = asset.extra_fields.get(property_name)
        if not item_property:
            item_property = item.properties.get(property_name)
        if not item_property:
            item_property = item.extra_fields.get(property_name)
        return item_property
    else:
        item_property = None
        for name, asset in item.assets.items():
            asset_property = get_property(item, property_name, name)
            if asset_property and item_property and asset_property != item_property:
                raise ValueError("Conflicting properties found in item over assets.")
            elif asset_property:
                item_property = asset_property
        return item_property


def mgrs_tiled_overlap_filter_coverage(
    items: list[MGRSTiledItem], roi: geom.Geometry, debug=False
) -> list[MGRSTiledItem]:
    """
    Filter Sentinel-2 items based on their coverage of the area of interest.
    Returns a list items required to cover the area of interest.
    """
    utm_center_easting = 500000
    criteria_helper = namedtuple(
        'criteria_helper', ['contains', 'negative_dist_utm_center', 'overlap_percentage', 'tile_id', 'geometry_union']
    )
    # remove all non overlapping geometries
    items = [item for item in items if geo.wgs84_intersects(item.geometry_odc_geometry, roi)]

    # group by mgrs
    mgrs_tiles = defaultdict(list)
    for item in items:
        mgrs_tiles[item.mgrs_tile].append(item)

    sort_criteria = []
    # overlaps = {}
    # utm_center_dist = {}
    for v in mgrs_tiles.values():
        proj = v[0].proj_code

        v_geometries = geom.unary_union([vv.geometry_odc_geometry for vv in v])
        # overlaps[v[0].mgrs_tile] = v_geometries.overlaps(roi)
        # utm_center_dist[v[0].mgrs_tile] = abs(v_geometries.to_crs(proj).centroid.points[0][0] - utm_center_easting)
        sort_criteria.append(
            criteria_helper(
                geo.wgs84_contains(v_geometries, roi),
                -abs(v_geometries.to_crs(proj).centroid.points[0][0] - utm_center_easting),
                v_geometries.intersection(roi).area / roi.area,
                v[0].mgrs_tile,
                v_geometries,
            )
        )

    sort_criteria = sorted(sort_criteria, reverse=True)

    if debug:
        print('roi: ', roi)
        for s in sort_criteria:
            print(s)

    if sort_criteria[0].contains:
        return mgrs_tiles[sort_criteria[0].tile_id]

    roi_coverage = 0.0
    iterative_shape = None
    return_mgrs = []
    current_iterative_shape = None
    for crit in sort_criteria:
        if not iterative_shape:
            current_iterative_shape = crit.geometry_union
        else:
            current_iterative_shape = geom.unary_union([crit.geometry_union, current_iterative_shape])
        current_roi_coverage = current_iterative_shape.intersection(roi).area / roi.area

        if current_roi_coverage > roi_coverage:
            roi_coverage = current_roi_coverage
            iterative_shape = current_iterative_shape
            return_mgrs.extend(mgrs_tiles[crit.tile_id])

    return return_mgrs


def no_overlap_filter_coverage(items, roi, min_overlap=None):
    """
    this method checks which crs is best to cover the roi based on input items max over lap of joined geometries
    sorts the items then in groups of crs with best coverage first
    min_overlap removes items which do not cover at least the min_overlap of the area

    Args:
        items (pystac.ItemCollection): input items
        roi (Geometry): region if interest to be covered
        min_overlap (float): minimum overlap in percent [0-1] of roi area to be covered by items

    Returns:
        list[pystac.Item]: filtered and sorted items
    """
    if not get_property(items[0], 'proj:code'):
        raise ValueError("Items do not have projection information, run extend stac item information first!")

    utm_center_easting = 500000
    criteria_helper = namedtuple(
        'criteria_helper', ['contains', 'dist_utm_center', 'overlap_percentage', 'proj_code', 'geometry_union']
    )
    # remove all non overlapping geometries
    if not min_overlap:
        items = [
            item
            for item in items
            if geo.wgs84_intersects(geom.Geometry(shapely.geometry.shape(item.geometry), '4326'), roi)
        ]
    else:
        items = [
            item
            for item in items
            if geo.wgs84_overlap_percentage(
                geom.Geometry(
                    shapely.geometry.shape(item.geometry),
                    '4326',
                ),
                roi,
            )
            >= min_overlap
        ]
    if not items:
        return []

    # group by utm
    utm_groups = defaultdict(list)
    for item in items:
        utm_groups[get_property(item, 'proj:code')].append(item)

    sort_criteria = []
    for v in utm_groups.values():
        proj = get_property(v[0], 'proj:code')

        for vv in v:
            for k in vv.get_assets():
                if get_property(vv, 'proj:code', k):
                    get_property(vv, 'proj:transform', k)
                    tr = get_property(vv, 'proj:transform', k)
                else:
                    tr = None
            if not tr:
                raise ValueError('No assets found with proj:code info!')
            if not any([tr[2] / tr[0] % 1.0 == 0.0, tr[5] / tr[4] % 1.0 == 0.0]):
                raise ValueError('Items of same UTM zone have sub-pixel grid offsets!')

        v_geometries = geom.unary_union([geom.Geometry(shapely.geometry.shape(vv.geometry), '4326') for vv in v])
        sort_criteria.append(
            criteria_helper(
                geo.wgs84_contains(v_geometries, roi),
                abs(v_geometries.to_crs(proj).centroid.points[0][0] - utm_center_easting),
                v_geometries.intersection(roi).area / roi.area,
                proj,
                v_geometries,
            )
        )

    sort_criteria = sorted(sort_criteria)
    return_groups = []
    for crit in sort_criteria:
        return_groups.extend(utm_groups[crit.proj_code])
    return return_groups


def filter_no_data_timesteps(data: xr.Dataset, indicator_variable: str | None = None) -> xr.Dataset:
    if not indicator_variable:
        coords_res = _get_coord_name_and_resolution(data)
        coarsest_axes = coords_res[max(coords_res)]
        for data_var in data.data_vars.values():
            if coarsest_axes[0] in data_var.dims:
                indicator_variable = data_var.name
                break

    mean_over_time = data[indicator_variable].mean(dim=coarsest_axes)
    no_data_keys = ['nodata_value', 'nodata']
    for k in no_data_keys:
        na_value = data[indicator_variable].attrs.get(k)
        if no_data_keys:
            break
    mask_over_time = np.where(mean_over_time != na_value)[0]
    return data.isel(time=mask_over_time)


def _get_coord_name_and_resolution(data: xr.Dataset) -> dict[float, str]:
    coord_names_resolution = defaultdict(list)
    for coord in data.coords.values():
        if coord.name.startswith('x') or coord.name.startswith('y'):
            coord_names_resolution[abs(coord.attrs['resolution'])].append(coord.name)
    return dict(coord_names_resolution)

"""
STAC related functions
"""

from dataclasses import dataclass
from functools import cached_property, total_ordering
from typing import Iterable

import odc.geo.geom as geom
import odc.stac
import pystac
import xarray as xr
from odc.geo.geobox import GeoBox


def geoboxes_from_assets(item: pystac.Item, asset_ids: str | Iterable[str] | None = None) -> dict[str, GeoBox]:
    """
    Parse geoboxes from the specified assets of a STAC item.

    Args:
        item (pystac.Item): The STAC item from which to extract geoboxes.
        asset_ids (str | Iterable[str], optional): The asset ID or list of asset IDs to extract geoboxes for.
            If None, geoboxes are returned for all assets in the item.

    Returns:
        dict[str, GeoBox]: A dictionary mapping asset IDs to their corresponding GeoBox objects.
            Only assets for which a geobox could be parsed are included.

    Raises:
        KeyError: If any of the specified asset_ids are unknown.

    Note:
        This functionality relies on the Projection STAC extension and Raster STAC extension to extract geospatial information from assets.
    """
    parsed_item = odc.stac.parse_item(item)
    asset_ids = parsed_item.collection.normalize_band_query(
        asset_ids
    )  # wraps single str in list, and makes None -> all_bands
    geoboxes = {}
    for asset_id in asset_ids:
        try:
            band_key = parsed_item.collection.band_key(asset_id)
        except (
            ValueError
        ):  # no band_key => asset was not recognized as raster band (e.g. due to missing proj attributes)
            continue
        band = parsed_item.bands[band_key]
        if band is not None and band.geobox is not None:
            geoboxes[asset_id] = band.geobox
    return geoboxes


def geobox_from_asset(item: pystac.Item, asset_id: str) -> GeoBox | None:
    """
    Parse the GeoBox associated with a specific asset from a STAC item.

    Args:
        item (pystac.Item): The STAC item containing the asset.
        asset_id (str): The identifier of the asset for which to retrieve the GeoBox.

    Returns:
        GeoBox | None: The GeoBox corresponding to the specified asset if available, otherwise None.

    Raises:
        KeyError: If the specified asset_id is unknown.

    Note:
        This functionality relies on the Projection STAC extension and Raster STAC extension to extract geospatial information from assets.
    """
    return geoboxes_from_assets(item, (asset_id,)).get(asset_id)


def group_assets_by_grid(item: pystac.Item) -> dict[GeoBox, list[str]]:
    """
    Groups asset names of a STAC item by their associated GeoBox.

    Given a pystac.Item, this function maps each asset to its corresponding GeoBox
    (using `geoboxes_from_assets`), then groups asset names by the common GeoBox they belong to.

    This function can be for instance be used to group variables of identical resolution together.

    Args:
        item (pystac.Item): The STAC item containing assets to be grouped.

    Returns:
        dict[GeoBox, Sequence[str]]: A dictionary mapping each GeoBox to a sequence of asset names that use the identical GeoBox
    """
    asset_to_gbox = geoboxes_from_assets(item)
    geoboxes = set(asset_to_gbox.values())
    geoboxes = sorted(geoboxes, key=lambda box: min(box.resolution.map(abs).xy))
    gbox_to_assets = {geobox: [] for geobox in geoboxes}
    for asset, gbox in asset_to_gbox.items():
        gbox_to_assets[gbox].append(asset)
    return gbox_to_assets


@dataclass(frozen=True)
@total_ordering
class EnclosingGeoboxResult:
    grid_box: GeoBox
    enclosing_box: GeoBox
    assets: list[str]

    @cached_property
    def absolute_resolution(self):
        return self.grid_box.resolution.map(abs)

    @cached_property
    def min_gsd(self) -> float:
        return min(self.absolute_resolution.xy)

    @cached_property
    def max_gsd(self) -> float:
        return max(self.absolute_resolution.xy)

    @cached_property
    def gsd(self) -> float | None:
        return self.min_gsd if self.min_gsd == self.max_gsd else None

    def __lt__(self, other: "EnclosingGeoboxResult"):
        self_res = (self.min_gsd, self.max_gsd)
        other_res = (other.min_gsd, other.max_gsd)
        if self_res != other_res:
            return self_res < other_res
        else:
            return self.assets < other.assets


def enclosing_geoboxes_per_grid(
    item: pystac.Item,
    geometry: geom.Geometry,
) -> list[EnclosingGeoboxResult]:
    """
    Finds the smallest GeoBox enclosing the given geometry for each grid/resolution present in the given STAC items.
    """
    parsed_item = odc.stac.parse_item(item)
    gbox_to_assets = group_assets_by_grid(item)

    results = []
    for grid_box, asset_names in gbox_to_assets.items():
        enclosing_box = odc.stac.output_geobox(
            [parsed_item],
            bands=asset_names,
            geopolygon=geometry,
        )
        result = EnclosingGeoboxResult(
            grid_box=grid_box,
            enclosing_box=enclosing_box,
            assets=asset_names,
        )
        results.append(result)

    results = sorted(results)  # sort in ascending gsd
    return results


def merge_datasets(datasets: list[xr.Dataset]) -> xr.Dataset:
    """
    Merge multiple xarray Datasets into a single Dataset with renamed coordinate variables.

    This function takes a list of xarray Datasets and merges them into one Dataset.
    If only one Dataset is provided, it is returned as-is.
    If multiple Datasets are provided, each Dataset's 'x' and 'y' coordinate variables
    are renamed to ensure uniqueness before merging.

    Args:
        datasets (list of xr.Dataset): List of xarray Datasets to merge.

    Returns:
        xr.Dataset: Merged Dataset containing all input Datasets with uniquely renamed coordinates.

    Raises:
        ValueError: If the input list is empty.
    """
    if not datasets:
        raise ValueError('No datasets provided')
    if len(datasets) == 1:
        return datasets[0]
    renamed_datasets = []
    for i, ds in enumerate(datasets):
        renamed_datasets.append(ds.rename({'x': f'x_{i}', 'y': f'y_{i}'}))
    return xr.merge(renamed_datasets)

"""
STAC related functions
"""

from typing import Iterable, Sequence

import odc.geo.geom as geom
import odc.stac
import pystac
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


def group_assets_by_grid(item: pystac.Item) -> dict[GeoBox, Sequence[str]]:
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
    gbox_to_assets = {geobox: [] for geobox in geoboxes}
    for asset, gbox in asset_to_gbox.items():
        gbox_to_assets[gbox].append(asset)
    return gbox_to_assets


def enclosing_geoboxes_per_grid(
    items: pystac.ItemCollection, geometry: geom.Geometry, crs: geom.MaybeCRS = None
) -> dict[GeoBox, GeoBox]:
    """
    Finds the smallest GeoBox enclosing the given geometry for each grid/resolution present in the given STAC items.
    """
    if len(items) == 0:
        raise ValueError('No items provided')

    parsed_items = list(odc.stac.parse_items(items))

    gbox_to_assets = group_assets_by_grid(items[0])  # assuming that each item is consistent
    gbox_to_enclosing_box = {}
    for gbox, asset_names in gbox_to_assets.items():
        gbox_to_enclosing_box[gbox] = odc.stac.output_geobox(
            parsed_items, bands=asset_names, geopolygon=geometry, crs=crs
        )

    return gbox_to_enclosing_box

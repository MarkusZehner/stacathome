"""
STAC related functions
"""

from typing import Iterable

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

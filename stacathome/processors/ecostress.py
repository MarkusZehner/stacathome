from collections import defaultdict
from pathlib import Path

import rasterio
from rasterio.env import Env
import pystac
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterExtension
from rio_stac.stac import (
    get_projection_info,
    get_raster_info,
)
from odc.geo import geom

from ..providers.common import BaseProvider
from ..stac import update_asset_exts
from .base import BaseProcessor
from .sentinel2 import S2Item, s2_pc_filter_coverage


class ECO_L2T_LSTEProcessor(BaseProcessor):
    '''
    process:
    -> request items, 
    -> filter per utm zune by name (or other metadata field)
    -> get exact proj for each utm zone
    -> filter items
    -> download data
    -> load to xarray 
    
    
    
    '''
    def filter_items(
        self, provider: BaseProvider, roi: geom.Geometry, items: pystac.ItemCollection
    ) -> pystac.ItemCollection:
        """

        Filter items based on the area of interest and the newest processing iteration.
        This function filters the items to ensure they cover the area of interest and selects the newest processing time for each item.
        Reoders the Items by the tile id, first item(s) corresponding to the tile closest to the roi.

        """
        extra_fields=items.extra_fields
        item_list = [item for item in items]
        item_list = ecostress_pc_filter_newest_processing_iteration(item_list)
        item_list = ecostress_update_from_cloud(item_list)
        item_list = [S2Item(item) for item in item_list]
        item_list = s2_pc_filter_coverage(item_list, roi)
        return pystac.ItemCollection(
            items=item_list,
            clone_items=False,
            extra_fields=extra_fields,
        )

def ecostress_pc_filter_newest_processing_iteration(items: list[pystac.Item]) -> list[pystac.Item]:
    """
    Returns the newest processing iteration of ECO_L2T_LSTE items using the processing iteration from the ID.
    The ID is expected to be in the format 'ECOv002_L2T_LSTE_28425_001_32MQE_20230711T175148_0710_01'.
    """
    filtered = {}
    for item in items:
        native_id = item.id
        base_name = native_id[:-8]
        product_iteration = native_id[-7:]
        if base_name not in filtered or product_iteration > filtered[base_name][0]:
            filtered[base_name] = (product_iteration, item)
    return [v[1] for v in filtered.values()]

def ecostress_update_from_cloud(items: list[pystac.Item]) -> list[pystac.Item]:
    """
    Gathers additional infromation by reading the metadata from the provider.
    To limit this, items are grouped by their UTM zone and all items of one zone are assumed to have the same proj and raster info.
    The ID is expected to be in the format 'ECOv002_L2T_LSTE_28425_001_32MQE_20230711T175148_0710_01'.
    """
    
    utm_dict = defaultdict(list)
    for item in items:
        utm_zone = item.id.split('_')[5]
        utm_dict[utm_zone].append(item)

    info_dict = {}
    cookie_file_urs = Path.home() / '.urs_cookies'
    if Path.is_file(cookie_file_urs):
        with Env(
            GDAL_HTTP_COOKIEFILE=cookie_file_urs,
            GDAL_DISABLE_READDIR_ON_OPEN='YES',
            GDAL_HTTP_TCP_KEEPALIVE = 'YES',
            GDAL_PAM_ENABLED='NO',          
            GDAL_MAX_DATASET_POOL_SIZE='1', 
            CPL_VSIL_CURL_USE_HEAD='NO',
            # CPL_CURL_VERBOSE='YES',
            VSI_CACHE='YES',
            VSI_CACHE_SIZE=100 * 1024 * 1024,
        ):
            for utm_zone, item_list in utm_dict.items():
                info_dict[utm_zone] = {}
                proj_info = {}
                for name, asset in item_list[0].get_assets().items():
                    info_dict[utm_zone][name] = {}
                    if proj_info == {}:
                        with rasterio.open(asset.href) as src_dst:
                            proj_info_set = get_projection_info(src_dst).items()
                            proj_info = {
                                f"proj:{name}": value
                                for name, value in proj_info_set if name in 
                                ['epsg', 'code', 'bbox', 'shape', 'transform']
                            }
                            if 'proj:epsg' in proj_info and not 'proj:code' in proj_info:
                                proj_info['proj:code'] = proj_info['proj:epsg']
                                del proj_info['proj:epsg']
                        # raster_info = {
                        #     "raster:bands": get_raster_info(src_dst, max_size=1024)
                        # }
                        # info_dict[utm_zone][name] = raster_info
                info_dict[utm_zone] = info_dict[utm_zone] | proj_info
                    

    items_with_exts = []                            
    for utm_zone, item_list in utm_dict.items():
        for item in item_list:
            ProjectionExtension.add_to(item)
            #RasterExtension.add_to(item)
            proj_ext = ProjectionExtension.ext(item)
            proj_ext.epsg = info_dict[utm_zone]['proj:code']
            proj_ext.bbox = info_dict[utm_zone]['proj:bbox']
            proj_ext.shape = info_dict[utm_zone]['proj:shape']
            proj_ext.transform = info_dict[utm_zone]['proj:transform']
            # for name, asset in item.get_assets().items():
            #     item.asset[name].extra_fields['raster:bands'] = info_dict[utm_zone][name]['raster:bands']
            item.properties['grid:code'] = utm_zone
            items_with_exts.append(item)
            
    return items_with_exts
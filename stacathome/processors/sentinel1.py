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

from .base import register_default_processor, SimpleProcessor
from ..providers import SimpleProvider
from .ecostress import handle_rasterio_env
from .common import no_overlap_filter_coverage, get_property


class Sentinel1OperaL2RTCProcessor(SimpleProcessor):

    def filter_items(
        self, provider: SimpleProvider, roi: geom.Geometry, items: pystac.ItemCollection
    ) -> pystac.ItemCollection:

        if not get_property(items[0], 'proj:code'):
            update_from_raster(items)

        items_filtered = no_overlap_filter_coverage(items, roi)

        return pystac.ItemCollection(
            items=items_filtered,
            clone_items=False,
            extra_fields=items.extra_fields,
        )


#     """
#     Processor for Sentinel-1 Opera L2 RTC data from EarthAccess/ASF.
#     - the data comes in 30 m burst(?) tiles oriented in utm zones
#     - decide if this should be used, the quality of the data is good, just lower resolution.
#     - aggregation per orbit_nr of same grid over time, but maybe it differs in pixels?
#     """


def update_from_raster(items: list[pystac.Item], proj=True, raster=False) -> list[pystac.Item]:
    """too slow for usage"""
    with Env(**handle_rasterio_env()):
        for i, _ in enumerate(items):
            proj_info = {}
            ProjectionExtension.add_to(items[i])
            RasterExtension.add_to(items[i])
            for name, asset in items[i].get_assets().items():
                # print(f"Processing asset: {name} - {asset.href}")
                with rasterio.open(asset.href) as src_dst:
                    if proj and not proj_info:
                        proj_info_set = get_projection_info(src_dst).items()
                        proj_info = {
                            f"proj:{pname}": value
                            for pname, value in proj_info_set
                            if pname in ['epsg', 'code', 'bbox', 'shape', 'transform']
                        }
                        if 'proj:epsg' in proj_info and not 'proj:code' in proj_info:
                            proj_info['proj:code'] = proj_info['proj:epsg']
                            del proj_info['proj:epsg']
                    if raster:
                        items[i].assets[name].extra_fields['raster:bands'] = get_raster_info(src_dst, max_size=1024)
            if proj and proj_info:
                proj_ext = ProjectionExtension.ext(items[i])
                proj_ext.epsg = proj_info['proj:code']
                proj_ext.bbox = proj_info['proj:bbox']
                proj_ext.shape = proj_info['proj:shape']
                proj_ext.transform = proj_info['proj:transform']
    return items


register_default_processor('earthaccess', 'sentinel-1-opera-l2rtc', Sentinel1OperaL2RTCProcessor)

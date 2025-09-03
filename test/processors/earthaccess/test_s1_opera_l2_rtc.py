import pytest
import shapely
from odc.geo.geobox import GeoBox

from stacathome.processors.common import MGRSTiledItem, mgrs_tiled_overlap_filter_coverage
from stacathome.processors.ecostress import ecostress_pc_filter_newest_processing_iteration, ecostress_update_from_cloud
from stacathome.providers import get_provider

NEAR_TILE_CORNER = shapely.Point(704800, 5895040) # EPSG:32632
NEAR_TILE_CORNER_SHIFT = shapely.Point(710800, 5901040) # EPSG:32632

def create_test_geobox(center_p, crs='EPSG:32632', resolution=10, size_box=500):
    """Create a geobox centered at the given point with a specified resolution and size."""
    bbox = (
        center_p.x - size_box,  # left
        center_p.y - size_box,  # bottom
        center_p.x + size_box,  # right
        center_p.y + size_box,  # top
    )  # 1km x 1km
    geobox = GeoBox.from_bbox(
        bbox=bbox,
        crs=crs,
        resolution=resolution,
        tight=True,
    )
    return geobox

class TestS1OPERAL2RTCProcessor:
    
    @pytest.mark.remote
    @pytest.mark.earthaccess
    def test_request_and_download(self, tmp_path):
        provider = get_provider('earthaccess')
        
        geobox_n = create_test_geobox(shapely.Point(723311,5895040), resolution=100, size_box=5000, crs='EPSG:32632')
        roi_n = geobox_n.footprint('EPSG:4326', buffer=0, npoints=4)

        geobox_s = create_test_geobox(shapely.Point(450000,9000000), resolution=100, size_box=5000, crs='EPSG:32733')
        roi_s = geobox_s.footprint('EPSG:4326', buffer=0, npoints=4)

        items_n = provider.request_items(
            collection='OPERA_L2_RTC-S1_V1',
            starttime='2023-07-10',
            endtime='2023-07-20',
            roi=roi_n,
            limit=4
        )

        items_s = provider.request_items(
            collection='OPERA_L2_RTC-S1_V1',
            starttime='2014-03-10',
            endtime='2025-08-30',
            roi=roi_s,
            limit=4
        )
        
        items_n = provider.load_granule(item=items_n, variables=['VH'], out_dir=tmp_path)
        items_s = provider.load_granule(item=items_s, variables=['VV'], out_dir=tmp_path)

        cube_n = provider.load_items(items_n, variables=['VH'], geobox=geobox_n)
        cube_s = provider.load_items(items_s, variables=['VV'], geobox=geobox_s)

        assert 'VH' in cube_n.data_vars
        assert 'VV' in cube_s.data_vars

        assert len(cube_n.time) == len(items_n)
        assert len(cube_s.time) == len(items_s)

    # def test_filtering(self):
    #     provider = get_provider('earthaccess')

    #     geobox = create_test_geobox(shapely.Point(723311,4635624), resolution=100, size_box=10000, crs='EPSG:32632')
    #     area_of_interest = geobox.footprint('EPSG:4326', buffer=0, npoints=4)

    #     geobox_small = create_test_geobox(shapely.Point(723311,4635624), resolution=100, size_box=100, crs='EPSG:32632')
    #     roi_small = geobox_small.footprint('EPSG:4326', buffer=0, npoints=4)

    #     items = provider.request_items(
    #         collection='OPERA_L2_RTC-S1_V1',
    #         # version ='002',
    #         starttime='2023-07-10',
    #         endtime='2023-07-20',
    #         roi=area_of_interest,
    #         limit=10
    #     )
    #     print(f"Retrieved {len(items)} items from provider.")
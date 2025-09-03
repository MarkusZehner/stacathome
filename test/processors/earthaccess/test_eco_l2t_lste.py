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


class TestECO_L2T_LSTEProcessor:
    
    @pytest.mark.remote
    @pytest.mark.earthaccess
    def test_filtering(self):
        provider = get_provider('earthaccess')

        geobox = create_test_geobox(shapely.Point(723311,4635624), resolution=100, size_box=10000, crs='EPSG:32632')
        area_of_interest = geobox.footprint('EPSG:4326', buffer=0, npoints=4)

        geobox_small = create_test_geobox(shapely.Point(723311,4635624), resolution=100, size_box=100, crs='EPSG:32632')
        roi_small = geobox_small.footprint('EPSG:4326', buffer=0, npoints=4)

        items = provider.request_items(
            collection='ECO_L2T_LSTE',
            version ='002',
            starttime='2023-07-10',
            endtime='2023-07-13',
            roi=area_of_interest,
            limit=10
        )
        
        assert len(items) == 4

        only_newer_processing = ecostress_pc_filter_newest_processing_iteration(items)
        assert len(only_newer_processing) == 2
        
        updated_items = ecostress_update_from_cloud(only_newer_processing)
        
        assert updated_items[0].properties.get('proj:code')
        
        updated_items = [MGRSTiledItem(item) for item in updated_items]

        coverage_filtered_items = mgrs_tiled_overlap_filter_coverage(updated_items, roi_small)
        assert len(coverage_filtered_items) == 1

        all_returns = [
            'ECOv002_L2T_LSTE_28442_005_32TQM_20230712T202458_0710_01',
            'ECOv002_L2T_LSTE_28442_005_33TTG_20230712T202458_0710_01',
            'ECOv002_L2T_LSTE_28442_005_33TTG_20230712T202458_0712_02',
            'ECOv002_L2T_LSTE_28442_005_32TQM_20230712T202458_0712_02',
 ]

        filter_processing_iteration = [
            'ECOv002_L2T_LSTE_28442_005_33TTG_20230712T202458_0712_02',
            'ECOv002_L2T_LSTE_28442_005_32TQM_20230712T202458_0712_02',
        ]

        filter_processing_coverage = [
            'ECOv002_L2T_LSTE_28442_005_33TTG_20230712T202458_0712_02',
        ]

        assert {i.id for i in items} == set(all_returns)
        assert {i.id for i in only_newer_processing} == set(filter_processing_iteration)
        assert {i.id for i in coverage_filtered_items} == set(filter_processing_coverage)


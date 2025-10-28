import pystac
import pytest
import shapely
from odc.geo.geom import Geometry
from stacathome.providers import EarthAccessProvider


class TestEarthAccessProvider:

    @pytest.mark.remote
    def test_get_items(self):
        start = '2023-01-01'
        end = '2023-01-02'
        roi = Geometry(shapely.box(11, 51, 11.5, 51.5), crs=4326)

        EXPECTED_ITEM_IDS = {
            "ECOv002_L2T_LSTE_25452_010_32UPB_20230101T031737_0710_01",
            "ECOv002_L2T_LSTE_25452_010_32UPC_20230101T031737_0710_01",
            "ECOv002_L2T_LSTE_25452_010_32UPB_20230101T031737_0712_02",
            "ECOv002_L2T_LSTE_25452_010_32UPC_20230101T031737_0712_02",
        }

        EXPECTED_ASSET_KEYS = {
            'EmisWB',
            'LST',
            'LST_err',
            'QC',
            'cloud',
            'height',
            'view_zenith',
            'water',
        }

        provider = EarthAccessProvider('earthaccess')

        item_col = provider.request_items(
            collection='ECO_L2T_LSTE',
            version='002',
            starttime=start,
            endtime=end,
            roi=roi,
        )
        assert isinstance(item_col, pystac.ItemCollection)
        assert len(item_col) == 4
        assert {item.id for item in item_col} == EXPECTED_ITEM_IDS
        assert set(item_col[0].get_assets().keys()) == EXPECTED_ASSET_KEYS

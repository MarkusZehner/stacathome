import pystac
import pytest
import shapely
from stacathome.providers import ASFProvider


class TestASFProvider:

    @pytest.mark.remote
    def test_get_items(self):
        start = '2023-01-01'
        end = '2023-01-02'
        roi = shapely.box(11, 51, 11.5, 51.5)

        EXPECTED_ITEM_IDS = {
            'SP_42291_D_008',
            }
        
        EXPECTED_ASSET_KEYS = {
            'QA',
            'XML',
            'HDF5',
            }
        provider = ASFProvider('asf')

        item_col = provider.request_items(
            collection='SMAP',
            starttime=start,
            endtime=end,
            roi=roi,
        )

        assert isinstance(item_col, pystac.ItemCollection)
        assert len(item_col) == 1
        assert {item.id for item in item_col} == EXPECTED_ITEM_IDS
        assert set(item_col[0].get_assets().keys()) == EXPECTED_ASSET_KEYS
import shapely
import pystac
import pytest

from stacathome.providers import ASFProvider



class TestASFProvider:

    @pytest.mark.remote
    def test_get_items(self):
        start = '2023-01-01'
        end = '2023-01-02'
        area_of_interest = shapely.box(11, 51, 11.5, 51.5)
        
        EXPECTED_ITEM_IDS = {
        }

        provider = ASFProvider()

        item_col = provider.request_items(
            collection='SMAP',
            starttime=start,
            endtime=end,
            area_of_interest=area_of_interest,
        )
        assert isinstance(item_col, pystac.ItemCollection)
        assert len(item_col) == 3
        assert set(item.id for item in item_col) == EXPECTED_ITEM_IDS
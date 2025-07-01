import shapely
import pystac
import pytest

from stacathome.providers import STACProvider, get_provider



class TestSTACProvider:

    @pytest.mark.remote
    def test_get_items_planetary(self):
        import planetary_computer

        time = '2023-01-01/2023-01-02'
        location = shapely.box(11, 51, 11.5, 51.5)
        
        EXPECTED_ITEM_IDS = {
            'S2B_MSIL2A_20230101T102339_R065_T32UPC_20240806T223544', 
            'S2B_MSIL2A_20230101T102339_R065_T32UPC_20230101T211104', 
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20240806T223544', 
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20230101T222806',
        }

        provider = STACProvider(
            url='https://planetarycomputer.microsoft.com/api/stac/v1',
            sign=planetary_computer.sign,
        )

        item_col = provider.request_items(
            collection=['sentinel-2-l2a'],
            request_time=time,
            request_place=location,
            max_retry=1,
        )
        assert isinstance(item_col, pystac.ItemCollection)
        assert len(item_col) == 4
        assert set(item.id for item in item_col) == EXPECTED_ITEM_IDS
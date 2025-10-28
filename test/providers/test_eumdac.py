import datetime

import pystac
import pytest
from stacathome.providers import EUMDACProvider


class TestEUMDACProvider:

    @pytest.mark.remote
    def test_get_items(self):
        date = datetime.datetime(2021, 2, 1)

        EXPECTED_ITEM_IDS = {
            'MSG4-SEVI-MSG15-0100-NA-20210201115743.291000000Z-NA',
            'MSG4-SEVI-MSG15-0100-NA-20210201114243.035000000Z-NA',
            'MSG4-SEVI-MSG15-0100-NA-20210201112742.777000000Z-NA',
            'MSG4-SEVI-MSG15-0100-NA-20210201111242.521000000Z-NA',
        }

        EXPECTED_ASSET_KEYS = {'url'}
        provider = EUMDACProvider('asf')

        item_col = provider.request_items(
            collection='EO:EUM:DAT:MSG:HRSEVIRI',
            starttime=date.replace(hour=11),
            endtime=date.replace(hour=12),
        )

        assert isinstance(item_col, pystac.ItemCollection)
        assert len(item_col) == 4
        assert {item.id for item in item_col} == EXPECTED_ITEM_IDS
        assert set(item_col[0].get_assets().keys()) == EXPECTED_ASSET_KEYS

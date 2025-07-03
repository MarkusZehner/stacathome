import importlib
import json

import pystac
import shapely

from stacathome.processors.sentinel2_rewrite import Sentinel2L2AProcessor
from stacathome.providers import get_provider

class TestSentinel2L2AProcessor:

    def test_filtering(self):
        provider = get_provider('planetary_computer')
        processor = Sentinel2L2AProcessor()
        area_of_interest=shapely.box(15.0, 51.0, 15.1, 51.1),

        resources = importlib.resources.files('stacathome')
        with resources.joinpath('resources/tests/s2_l2a_items.json').open('rb') as f:
            items = pystac.ItemCollection.from_dict(json.load(f))

        filtered_items = processor.filter_items(provider, area_of_interest, items)
        assert len(filtered_items) > 0
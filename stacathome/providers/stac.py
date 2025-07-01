import time
from datetime import datetime
from functools import partial
from typing import Callable

import odc
import planetary_computer
import pystac
import pystac_client
import shapely
from rasterio.errors import RasterioIOError, WarpOperationError

from stacathome.generic_utils import download_assets_parallel
from .common import BaseProvider, register_provider


class STACProvider(BaseProvider):
    def __init__(self, url: str, sign: Callable):
        self.url = url
        self.sign = sign
        self.client = pystac_client.Client.open(self.url)

    def _request_items(
        self,
        collection: str,
        starttime: datetime,
        endtime: datetime,
        area_of_interest: shapely.Geometry = None,
        limit: int = None,
        **kwargs,
    ) -> pystac.ItemCollection:
        items = self.client.search(
            collections=[collection],
            datetime=(starttime, endtime),
            intersects=area_of_interest,
            limit=limit,
            **kwargs,
        ).item_collection()
        if items is None:
            raise ValueError("Failed to get data from the API")
        return items

    def download_granules_to_file(self, href_path_tuples: list[tuple]):
        download_assets_parallel(href_path_tuples, signer=self.sign)

    def download_cube(self, parameters):
        parameters.setdefault("patch_url", self.sign)
        data = None
        for i in range(5):
            try:
                data = odc.stac.load(**parameters)
                break
            except (WarpOperationError, RasterioIOError):
                print(f"Error creating cube: retry {i}", flush=True)
                time.sleep(5)
        if data is None:
            raise ValueError("Failed to create cube")
        return data


register_provider(
    'planetary_computer',
    partial(STACProvider, url='https://planetarycomputer.microsoft.com/api/stac/v1', sign=planetary_computer.sign),
)

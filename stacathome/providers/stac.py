import logging
import time
from functools import partial
from typing import Callable

import odc
import planetary_computer
import pystac_client
from pystac_client.exceptions import APIError
from rasterio.errors import RasterioIOError, WarpOperationError

from stacathome.generic_utils import download_assets_parallel
from .common import BaseProvider, register_provider


class STACProvider(BaseProvider):
    def __init__(
        self,
        url: str,
        sign: Callable,
        **kwargs,
    ):
        self.url = url
        self.sign = sign
        self.client = pystac_client.Client.open(self.url)
        self.extra_attributes = kwargs

    def request_items(
        self, collection: list[str], request_time: str, request_place: any = None, max_retry: int = 5, **kwargs
    ):
        if isinstance(collection, str):
            collection = [collection]
        items = None
        for _ in range(max_retry):
            try:
                items = self.client.search(
                    collections=collection, datetime=request_time, intersects=request_place, **kwargs
                ).item_collection()
                break
            except APIError as e:
                logging.warning(f"APIError: Retrying because of {e}")
                if "429" in str(e):
                    # Too many requests, wait and retry
                    time.sleep(3)
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

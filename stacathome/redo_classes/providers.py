import time
import logging

import planetary_computer as pc
from odc.stac import load
from pystac_client import Client
from pystac_client.exceptions import APIError
from rasterio.errors import RasterioIOError, WarpOperationError


from stacathome.asset_specs_class import download_assets_parallel
# from base import STACItemProcessor

logging.basicConfig(
    level=logging.INFO,  # Set to WARNING or ERROR in production
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class STACProvider():
    def __init__(
        self,
        url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        sign: callable = pc.sign,
        **kwargs
    ):
        self.url = url
        self.sign = sign
        self.client = Client.open(self.url)
        self.extra_attributes = kwargs
        # self.query = None

        # possible to add checks if cql2 filter, which collections are present...
        # should the classes have a default pipe-through if all their functions if all kwargs are matching?

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    def request_items(
        self,
        collection: list[str],
        request_time: str,
        request_place: any = None,
        max_retry: int = 5,
        **kwargs
    ):
        if isinstance(collection, str):
            collection = [collection]
        items = None
        for _ in range(max_retry):
            try:
                items = self.client.search(collections=collection,
                                           datetime=request_time,
                                           intersects=request_place,
                                           ** kwargs).item_collection()
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
                data = load(**parameters).load()
                break
            except (WarpOperationError, RasterioIOError):
                print(f"Error creating cube: retry {i}", flush=True)
                time.sleep(5)
        if data is None:
            raise ValueError("Failed to create cube")
        return data

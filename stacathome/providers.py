import time
import logging

import planetary_computer as pc
from odc.stac import load
from pystac_client import Client
from pystac_client.exceptions import APIError
from rasterio.errors import RasterioIOError, WarpOperationError

import asf_search
from asf_search.download import download_urls

import earthaccess


from stacathome.generic_utils import download_assets_parallel

logging.basicConfig(
    level=logging.INFO,  # Set to WARNING or ERROR in production
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class BaseProvider():
    pass


class EarthAccessProvider():
    def __init__(
            self
    ):
        earthaccess.login(persist=True)

    def request_items(self, request_time: str, request_place, **kwargs):
        bounds = request_place.bounds
        if '/' in request_time:
            start_time, end_time = request_time.split('/')
        else:
            raise ValueError('Earthaccess (probably) requires start and end time in form of yyyy-mm-dd/yyyy-mm-dd')
            start_time = request_time
            end_time = None

        granules = earthaccess.search_data(
            temporal=(start_time, end_time),
            bounding_box=bounds,
            **kwargs
        )
        return granules

    @classmethod
    def download_from_earthaccess(cls, granules, local_path, threads, **kwargs):
        earthaccess.download(
            granules,
            local_path=local_path,
            threads=threads,
            **kwargs)

    def create_cube(self, parameters):
        data = load(**parameters)
        if data is None:
            raise ValueError("Failed to create cube")
        return data


class ASFProvider():

    @staticmethod
    def request_items(request_time, request_place, **kwargs):
        # (collection=collection,
        # request_time=req_time,
        # request_place=self.request_place,
        # **kwargs)
        wkt = request_place.wkt
        if '/' in request_time:
            start_time, end_time = request_time.split('/')
        else:
            raise ValueError('ASF (probably) requires start and end time in form of yyyy-mm-dd/yyyy-mm-dd')
            start_time = request_time
            end_time = None

        print(wkt)

        results = asf_search.search(
            start=start_time,
            end=end_time,
            intersectsWith=wkt,
            **kwargs,
        )
        return results

    @staticmethod
    def download_from_asf(urls, path, **kwargs):
        download_urls(urls, path=path, **kwargs)

    def create_cube(self, parameters):
        data = load(**parameters)
        if data is None:
            raise ValueError("Failed to create cube")
        return data


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

    # def download_cube(self, parameters, split_by: int = None):
    #     parameters.setdefault("patch_url", self.sign)

    #     if split_by is not None and len(parameters['items']) > split_by:
    #         p_copy = parameters.copy()
    #         items = p_copy['items']
    #         print('len items', len(items))
    #         items = [items[i:i + split_by] for i in range(0, len(items), split_by)]
    #         data = []
    #         for item in items:
    #             p_copy['items'] = item
    #             data.append(self.__download_cube(p_copy))
    #             print(data)
    #         data = xr.concat(data, dim="time")
    #         print(data)
    #     else:
    #         data = self.__download_cube(parameters)

    #     return data

    def download_cube(self, parameters):
        parameters.setdefault("patch_url", self.sign)
        data = None
        for i in range(5):
            try:
                data = load(**parameters)  # .load()
                break
            except (WarpOperationError, RasterioIOError):
                print(f"Error creating cube: retry {i}", flush=True)
                time.sleep(5)
        if data is None:
            raise ValueError("Failed to create cube")
        return data

'''
how to: https://user.eumetsat.int/resources/user-guides/eumetsat-data-access-client-eumdac-guide

create account at : https://api.eumetsat.int
get keys at: https://api.eumetsat.int/api-key/

$ uv run eumdac set-credentials ConsumerKey ConsumerSecret

get license at : https://user.eumetsat.int/profile?activeTab=data-licenses

It is possible to generate an API access token by calling the token API service using the credentials provided above. Below the cURL command:

curl -k -d "grant_type=client_credentials" \
-H "Authorization: Basic Base64(consumer-key:consumer-secret)" \
https://api.eumetsat.int/token

It should be added in the http header of each API call as shown in the following sample cURL command:

curl -k \
-H "Authorization: Bearer 61834514-36d4-3949-8633-317a18a8143f" \
<api-endpoint>

install epct for local data tailor: 
micromamba create -p /.../datatailor python=3.9 -c eumetsat epct_restapi epct_webui epct_plugin_gis msg-gdal-driver

'''

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import os
import re
import shutil
import time

import shapely
import pystac
import eumdac
import eumdac.cli
import eumdac.product
from eumdac.request import get
from eumdac.errors import EumdacError, eumdac_raise_for_status
from requests.exceptions import HTTPError

# from odc.stac import load
from odc.geo.geom import Geometry

from .common import BaseProvider, register_provider


class EUMDACProvider(BaseProvider):
    def __init__(self, provider_name: str):
        super().__init__(provider_name)
        self.token = eumdac.AccessToken(eumdac.cli.load_credentials())
        try:
            str(self.token)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                raise ValueError(
                    'Set up eumdac credentials by running $ (uv run) eumdac set-credentials ConsumerKey ConsumerSecret, '
                    'with API keys from https://api.eumetsat.int/api-key/'
                ) from e
            raise

    def available_collections(self) -> list[str]:
        return [str(i) for i in eumdac.DataStore(self.token).collections]

    def _request_items(
        self,
        collection: str,
        starttime: datetime,
        endtime: datetime,
        roi: Geometry = None,
        limit: int = None,
        **kwargs,
    ) -> pystac.ItemCollection:

        if roi:
            roi = roi.wkt
        datastore = eumdac.DataStore(self.token)
        products = datastore.get_collection(collection).search(
            dtstart=starttime,
            dtend=endtime,
            geo=roi,
            # limit=limit,
            # set='brief',  # could be faster to use with datastore.get_product(id, collection)
        )

        return self.to_itemcollection(list(products))

    def to_itemcollection(self, granules: list[eumdac.product.Product]) -> pystac.ItemCollection:
        items = [self.create_item(granule) for granule in granules]
        item_collection = pystac.item_collection.ItemCollection(items, clone_items=False)

        return item_collection

    def create_item(self, granule: eumdac.product.Product) -> pystac.Item:
        """
        # this should be moved out to .stac
        Create a STAC item from a Granule object.
        Args:
            granule (eumdac.product.Product): The granule to convert into a STAC item.
        Returns:
            pystac.Item: The created STAC item.
        """
        g_dict = granule.__dict__
        item_id = g_dict['_id']
        date = g_dict['_browse_properties']['date']

        item_datetime = None
        if '/' in date:
            item_start_datetime, item_end_datetime = date.split('/')
        else:
            item_datetime = datetime.fromisoformat(date.replace("Z", "+00:00"))

        if item_start_datetime:
            item_start_datetime = datetime.fromisoformat(item_start_datetime.replace("Z", "+00:00"))
        if item_end_datetime:
            item_end_datetime = datetime.fromisoformat(item_end_datetime.replace("Z", "+00:00"))

        item_geometry = g_dict['_geometry']
        item_bbox = None
        if item_geometry:
            item_bbox = list(shapely.from_geojson(str(item_geometry).replace("'", '"')).bounds)

        assets = {
            'url': pystac.Asset(
                href=granule.datastore.urls.get(
                    "datastore",
                    "download product",
                    vars={"collection_id": str(granule.collection), "product_id": str(granule)},
                ),
            )
        }

        del g_dict['datastore']
        del g_dict['collection']

        item = pystac.Item(
            id=item_id,
            datetime=item_datetime,
            start_datetime=item_start_datetime,
            end_datetime=item_end_datetime,
            geometry=item_geometry,
            bbox=item_bbox,
            properties={'original_result': g_dict},  # needed for download? -> could be just href
            assets=assets,
        )

        item.validate()

        return item

    def download_granule(self, itemcollection, local_path=None, threads=None, max_retries=3, **kwargs):
        local_path = local_path if local_path else ''
        os.makedirs(local_path, exist_ok=True)

        def download_with_retry(item):
            for attempt in range(1, max_retries + 1):
                destination_file_name = os.path.join(local_path, item.id + '.zip')
                try:
                    download_item(destination_file_name, item, self.token)
                    return  # success
                except Exception as e:
                    print(f"[{item.id}] Attempt {attempt} failed: {e}")
                    os.remove(destination_file_name)
                    if attempt < max_retries:
                        wait = 2**attempt  # exponential backoff: 2s, 4s, 8s...
                        print(f"[{item.id}] Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"[{item.id}] Failed after {max_retries} attempts.")

        if threads and threads > 1:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = [executor.submit(download_with_retry, item) for item in itemcollection]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        pass  # Already logged inside `download_wrapper`
        else:
            for item in itemcollection:
                download_with_retry(item)


def download_item(destination_file_name, item, token):
    # print(f"Downloading {item.id} in thread {threading.current_thread().name}")
    if os.path.isfile(destination_file_name):
        print(f'File {destination_file_name} already exists.')
        return

    url = item.get_assets().get('url', {}).href
    auth = token.auth
    params = None
    headers = eumdac.common.headers.copy()

    with get(url, auth=auth, params=params, stream=True, headers=headers) as response:
        eumdac_raise_for_status(
            f"Could not download Product {item.id}",
            response,
            ProductError,
        )
        match = re.compile(r'filename="(.*?)"').search(response.headers["Content-Disposition"])
        filename = match.group(1)
        response.raw.name = filename
        response.raw.decode_content = True

        with open(destination_file_name, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)


class ProductError(EumdacError):
    """Errors related to products"""


#     def create_cube(self, parameters):
#         data = load(**parameters)
#         if data is None:
#             raise ValueError("Failed to create cube")
#         return data


register_provider('eumdac', EUMDACProvider)

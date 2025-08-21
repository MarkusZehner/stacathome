import os
from datetime import datetime
from functools import partial
from typing import Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import odc
import odc.stac
from odc.geo.geom import Geometry
import planetary_computer
import pystac
import pystac_client
import xarray as xr
from odc.geo.geobox import GeoBox
from urllib.request import urlretrieve

from stacathome.metadata import CollectionMetadata, Variable
from .common import BaseProvider, register_provider


class STACProvider(BaseProvider):

    def __init__(self, provider_name: str, url: str, sign: Callable):
        super().__init__(provider_name)
        self.url = url
        self.sign = sign
        self.client = pystac_client.Client.open(self.url)

    def _collection_client(self, collection: str) -> pystac_client.CollectionClient:
        # get_collection() is not consistence here wrt errors. Hence we check it ourself.
        if not self.has_collection(collection):
            raise KeyError(f'Collection {collection} not found')
        return self.client.get_collection(collection)

    def available_collections(self) -> list[str]:
        return [col.id for col in self.client.get_collections()]

    def get_metadata(self, collection) -> CollectionMetadata:
        stac_collection = self.client.get_collection(collection)
        item_assets = stac_collection.item_assets

        variables = []
        for name, asset_def in item_assets.items():
            # some providers/collections are not STAC v1 conform and don't provide roles
            # e.g. planetara-computer/alos-palsar-mosaic; here there exists a "role" top-level property
            if asset_def.roles:
                roles = asset_def.roles
            elif 'role' in asset_def.properties:
                roles = [asset_def.properties['role']]
            else:
                roles = []

            if 'data' not in roles:
                continue

            var = Variable(name)

            longname = asset_def.title
            description = asset_def.description
            dtype = None

            spatial_resolution = asset_def.properties.get('gsd')
            nodata_value = None
            scale = None
            offset = None
            unit = None
            center_wavelength = None
            full_width_half_max = None

            if 'eo:bands' in asset_def.properties:
                eo_bands: list = asset_def.properties['eo:bands']
                if len(eo_bands) == 1:
                    band = eo_bands[0]
                    description = band.get('description')
                    longname = band.get('common_name')
                    center_wavelength = band.get('center_wavelength')
                    full_width_half_max = band.get('full_width_half_max')

            if 'raster:bands' in asset_def.properties:
                raster_bands = asset_def.properties['raster:bands']
                if len(raster_bands) == 1:
                    band = raster_bands[0]
                    scale = band.get('scale')
                    nodata_value = band.get('nodata')
                    offset = band.get('offset')
                    dtype = band.get('data_type')
                    spatial_resolution = band.get('spatial_resolution')
                    unit = band.get('unit')

            var = Variable(
                name=name,
                longname=longname,
                description=description,
                roles=roles,
                dtype=dtype,
                spatial_resolution=spatial_resolution,
                nodata_value=nodata_value,
                scale=scale,
                offset=offset,
                unit=unit,
                center_wavelength=center_wavelength,
                full_width_half_max=full_width_half_max,
            )
            variables.append(var)

        return CollectionMetadata(*variables) if variables else None

    def get_item(self, collection: str, item_id: str) -> pystac.Item | None:
        col_client = self._collection_client(collection)
        return next(col_client.get_items(item_id), None)

    def _request_items(
        self,
        collection: str,
        starttime: datetime,
        endtime: datetime,
        roi: Geometry = None,
        limit: int = None,
        **kwargs,
    ) -> pystac.ItemCollection:
        items = self.client.search(
            collections=[collection],
            datetime=(starttime, endtime),
            intersects=roi,
            limit=limit,
            **kwargs,
        ).item_collection()
        if items is None:
            raise ValueError("Failed to get data from the API")
        return items

    def load_items(
        self, items: pystac.ItemCollection, geobox: GeoBox | None = None, variables: list[str] = None, **kwargs
    ) -> xr.Dataset:
        if not items:
            raise ValueError('No items provided for loading.')

        variables = set(variables) if variables else None
        groupby = kwargs.pop('groupby', 'id')

        # we do not wan to sort before load, as the item order from filter gives preferred projection
        # sorted_items = False
        # for datetime_key in ['start_time', 'start_datetime', 'datetime']:
        #     if all(datetime_key in item.properties for item in items):
        #         items = sorted(items, key=lambda x: x.properties.get(datetime_key))
        #         sorted_items = True
        #         break

        # if not sorted_items:
        #     raise ValueError('Inconsistent start_time/datetime info in items, check the ItemCollection!')

        data = odc.stac.load(
            items=items,
            bands=variables,
            patch_url=self.sign,
            geobox=geobox,
            groupby=groupby,
            # This is important for the filtering to be used!
            # By default items are sorted by time, id within each group to make pixel fusing order deterministic.
            # Setting this flag to True will instead keep items within each group in the same order as supplied,
            # so that one can implement arbitrary priority for pixel overlap cases.
            preserve_original_order=True,
            **kwargs,
        )
        # sort data by time
        data = data.sortby('time')
        return data

    def load_granule(
        self,
        out_dir: Path | str,
        item: pystac.Item | list[pystac.Item],
        variables: list[str],
        threads: int = 8,
        **kwargs,
    ) -> bytes:
        if isinstance(item, pystac.Item):
            item = [item]

        href_path_tuples = [
            (asset.href, os.path.join(out_dir, i.id, os.path.basename(asset.href)))
            for i in item
            for v, asset in i.get_assets().items()
            if v in variables
        ]
        return self._download_assets_parallel(href_path_tuples, threads)

    def _get_asset(
        self,
        href: str,
        save_path: Path,
    ):
        """
        Get one asset from a given href and save it to the specified path.
        This function will create the necessary directories if they do not exist,
        and will skip downloading if the file already exists.
        It also handles cleanup in case of an interruption during the download.

        Parameters:
        ----------
        href : str
            The URL of the asset to download.
        save_path : Path
            The local path where the asset should be saved.

        Returns:
        -------
        None

        Raises:
        -------
        Exception: If there is an error during the download process.
        KeyboardInterrupt: If the download is interrupted by the user.
        SystemExit: If the download is interrupted by a system exit.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            return
        try:
            urlretrieve(self.sign(href), save_path)
        except (KeyboardInterrupt, SystemExit):
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception as e:
                    print(f"Error during cleanup of file {save_path}:", e)
        except Exception as e:
            print(f"Error downloading {href}:", e)

    def _download_assets_parallel(
        self,
        asset_list: tuple[str, str],
        threads: int = 4,
    ):
        """
        Download a list of assets in parallel using a thread pool executor.
        This function will create a partial function with the signer and then use
        a thread pool to download each asset concurrently.

        Parameters:
        ----------
        asset_list : list of tuples
            A list where each tuple contains the href and the save path for the asset.
            Example: [(href1, save_path1), (href2, save_path2), ...]
        threads : int, default 4
            The maximum number of worker threads to use for downloading.

        Returns:
        -------
        None
        """
        with ThreadPoolExecutor(max_workers=threads) as executor:
            executor.map(lambda args: self._get_asset(*args), asset_list)


_planetary = partial(
    STACProvider, url='https://planetarycomputer.microsoft.com/api/stac/v1', sign=planetary_computer.sign
)
register_provider('planetary_computer', _planetary)

## get access to cdse via:
# https://documentation.dataspace.copernicus.eu/APIs/S3.html

## to use simply add this to environment before load_items call:
# import os
# os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"
# os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
# os.environ["AWS_ACCESS_KEY_ID"] = ""  # !
# os.environ["AWS_SECRET_ACCESS_KEY"] = ""  # !
# os.environ["AWS_HTTPS"] = "YES"
# os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
# os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
_cdse = partial(
    STACProvider,
    url='https://stac.dataspace.copernicus.eu/v1/',
    sign=None,
)
register_provider('cdse', _cdse)

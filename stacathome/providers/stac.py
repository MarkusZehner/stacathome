from datetime import datetime
from functools import partial
from typing import Callable

import odc
import odc.stac
from odc.geo.geom import Geometry
import planetary_computer
import pystac
import pystac_client
import xarray as xr
from odc.geo.geobox import GeoBox

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
        self, items: pystac.ItemCollection, geobox: GeoBox | None = None, variables=None, **kwargs
    ) -> xr.Dataset:
        if not items:
            raise ValueError('No items provided for loading.')

        variables = set(variables) if variables else None
        groupby = kwargs.pop('groupby', 'id')

        sorted_items = False
        for datetime_key in ['start_time', 'start_datetime', 'datetime']:
            if all(datetime_key in item.properties for item in items):
                items = sorted(items, key=lambda x: x.properties.get(datetime_key))
                sorted_items = True
                break

        if not sorted_items:
            raise ValueError('Inconsistent start_time/datetime info in items, check the ItemCollection!')
                
        data = odc.stac.load(
            items=items,
            bands=variables,
            patch_url=self.sign,
            geobox=geobox,
            groupby=groupby,
            **kwargs,
        )
        return data

    def load_granule(self, item: pystac.Item, **kwargs) -> bytes:
        raise NotImplementedError


_planetary = partial(
    STACProvider, url='https://planetarycomputer.microsoft.com/api/stac/v1', sign=planetary_computer.sign
)
register_provider('planetary_computer', _planetary)

# get access to cdse via:
# https://documentation.dataspace.copernicus.eu/APIs/S3.html
_cdse = partial(
    STACProvider, url='https://stac.dataspace.copernicus.eu/v1/', sign=None,
)
register_provider('cdse', _cdse)

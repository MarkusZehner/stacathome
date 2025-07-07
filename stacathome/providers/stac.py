from datetime import datetime
from functools import partial
from typing import Callable

import odc
import odc.stac
import planetary_computer
import pystac
import pystac_client
import shapely
import xarray as xr
from odc.geo.geobox import GeoBox

from stacathome.metadata import CollectionMetadata, Variable
from .common import BaseProvider, register_provider


class STACProvider(BaseProvider):

    def __init__(self, url: str, sign: Callable):
        self.url = url
        self.sign = sign
        self.client = pystac_client.Client.open(self.url)

    def get_metadata(self, collection) -> CollectionMetadata:
        stac_collection = self.client.get_collection(collection)
        item_assets = stac_collection.item_assets

        variables = []
        for name, asset_def in item_assets.items():
            if 'data' not in asset_def.roles:
                continue

            var = Variable(name)

            longname = asset_def.title
            description = asset_def.description
            roles = list(asset_def.roles)
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

    def load_items(self, items: pystac.ItemCollection, geobox: GeoBox | None = None, **kwargs) -> xr.Dataset:
        groupby = kwargs.pop('groupby', 'id')
        data = odc.stac.load(
            items=items,
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

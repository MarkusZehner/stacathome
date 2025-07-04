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

from datetime import datetime
from typing import Iterable

import pystac
import shapely
import xarray as xr
from odc.geo.geobox import GeoBox

from stacathome.processors import BaseProcessor, get_default_processor
from stacathome.providers import BaseProvider, get_provider


__all__ = [
    'search_items',
    'search_items_geoboxed',
    'load',
    'load_geoboxed',
]


def search_items(
    provider_name: str,
    collection: str,
    area_of_interest: shapely.Geometry,
    starttime: datetime,
    endtime: datetime,
    processor: BaseProcessor = None,
    no_default_processor: bool = False,
):
    provider = get_provider(provider_name)
    if processor is None and not no_default_processor:
        processor = get_default_processor(provider_name, collection)
    if processor is None:
        processor = BaseProcessor()

    items = provider.request_items(
        collection=collection,
        starttime=starttime,
        endtime=endtime,
        area_of_interest=area_of_interest,
    )
    items = processor.filter_items(provider, area_of_interest, items)
    return items


def search_items_geoboxed(
    provider_name: str,
    collection: str,
    geobox: GeoBox,
    starttime: datetime,
    endtime: datetime,
    processor: BaseProcessor = None,
    no_default_processor: bool = False,
):
    area_of_interest = geobox.footprint('EPSG:4326', buffer=10, npoints=4)
    return search_items(
        provider_name=provider_name,
        collection=collection,
        area_of_interest=area_of_interest,
        starttime=starttime,
        endtime=endtime,
        processor=processor,
        no_default_processor=no_default_processor,
    )


def load(
    provider_name: str,
    collection: str,
    area_of_interest: shapely.Geometry,
    starttime: datetime,
    endtime: datetime,
    processor: BaseProcessor = None,
    no_default_processor: bool = False,
) -> tuple[pystac.ItemCollection, xr.Dataset]:
    provider = get_provider(provider_name)
    if processor is None and not no_default_processor:
        processor = get_default_processor(provider_name, collection)
    if processor is None:
        processor = BaseProcessor()

    items = provider.request_items(
        collection=collection,
        starttime=starttime,
        endtime=endtime,
        area_of_interest=area_of_interest,
    )

    items = processor.filter_items(provider, area_of_interest, items)
    data = processor.load_items(provider, area_of_interest, items)
    data = processor.postprocess_data(provider, area_of_interest, data)

    return items, data


def load_geoboxed(
    provider_name: str,
    collection: str,
    geobox: GeoBox,
    starttime: datetime,
    endtime: datetime,
    processor: BaseProcessor = None,
    no_default_processor: bool = False,
) -> tuple[pystac.ItemCollection, xr.Dataset]:
    provider = get_provider(provider_name)
    if processor is None and not no_default_processor:
        processor = get_default_processor(provider_name, collection)
    if processor is None:
        processor = BaseProcessor()

    area_of_interest = geobox.footprint('EPSG:4326', buffer=10, npoints=4)

    items = provider.request_items(
        collection=collection,
        starttime=starttime,
        endtime=endtime,
        area_of_interest=area_of_interest,
    )

    items = processor.filter_items(provider, area_of_interest, items)
    data = processor.load_items_geoboxed(provider, geobox, items)
    data = processor.postprocess_data(provider, area_of_interest, data)

    return items, data

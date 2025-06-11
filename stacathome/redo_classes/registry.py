from pystac import Item
from asf_search import Products
from stacathome.redo_classes.processors import (Sentinel2L2AProcessor,
                                                Modis13Q1Processor,
                                                ESAWorldCoverProcessor,
                                                Sentinel1RTCProcessor,
                                                Sentinel3SynergyProcessor,
                                                OPERASentinel1RTCProcessor)

PROCESSOR_REGISTRY_STAC = {
    "sentinel-2-l2a": Sentinel2L2AProcessor,
    "modis-13Q1-061": Modis13Q1Processor,
    "esa-worldcover": ESAWorldCoverProcessor,
    "sentinel-1-rtc": Sentinel1RTCProcessor,
    "sentinel-3-synergy-syn-l2-netcdf": Sentinel3SynergyProcessor,
}

PROCESSOR_REGISTRY_ASF = {
    "OPERAS1Product": OPERASentinel1RTCProcessor,
}

PROCESSOR_REGISTRY = PROCESSOR_REGISTRY_STAC | PROCESSOR_REGISTRY_ASF


def get_supported_bands(dataset_key: str):
    return PROCESSOR_REGISTRY[dataset_key].get_supported_bands()


def get_tilename_key(dataset_key: str):
    return PROCESSOR_REGISTRY[dataset_key].get_tilename_key()


def get_processor(item):
    """
    this will return either the class (by string indicating the collection) 
    or an instantiated object (by item or result of search)

    Args:
        item (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if isinstance(item, Item):
        proc_cls = PROCESSOR_REGISTRY.get(item.collection_id, None)
    elif isinstance(item, Products.OPERAS1Product):
        proc_cls = PROCESSOR_REGISTRY.get(item.get_classname(), None)
    elif isinstance(item, str):
        proc_cls = PROCESSOR_REGISTRY.get(item, None)
    else:
        raise ValueError(f'item {item} not supported')

    if proc_cls is None:
        raise ValueError(f'item {item} has no registered processor')
    return proc_cls if isinstance(item, str) else proc_cls(item)


def request_items(collection, request_time, request_place, **kwargs):
    return PROCESSOR_REGISTRY[collection].request_items(collection, request_time, request_place, **kwargs)


def request_items_tile(collection, request_time, tile, **kwargs):
    return PROCESSOR_REGISTRY[collection].request_items_tile(
        collection, request_time, get_tilename_key(collection), tile, **kwargs)

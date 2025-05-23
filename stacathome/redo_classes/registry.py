from stacathome.redo_classes.processors import (Sentinel2L2AProcessor,
                                                Modis13Q1Processor,
                                                ESAWorldCoverProcessor,
                                                Sentinel1RTCProcessor,
                                                Sentinel3SynergyProcessor)

PROCESSOR_REGISTRY = {
    "sentinel-2-l2a": Sentinel2L2AProcessor,
    "modis-13Q1-061": Modis13Q1Processor,
    "esa-worldcover": ESAWorldCoverProcessor,
    "sentinel-1-rtc": Sentinel1RTCProcessor,
    "sentinel-3-synergy-syn-l2-netcdf": Sentinel3SynergyProcessor,
}


def get_supported_bands(dataset_key: str):
    return PROCESSOR_REGISTRY[dataset_key].get_supported_bands()


def get_tilename_key(dataset_key: str):
    return PROCESSOR_REGISTRY[dataset_key].get_tilename_key()


def get_processor(item):
    proc_cls = PROCESSOR_REGISTRY[item.collection_id]
    return proc_cls(item) if item else proc_cls

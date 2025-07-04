# from .ecostress import ECOL2TLSTEProcessor
# from .landsat import LandsatC2L2Processor
# from .modis import Modis13Q1Processor
# from .sentinel1 import OPERASentinel1RTCProcessor, Sentinel1RTCProcessor
# from .sentinel2 import Sentinel2L2AProcessor
# from .sentinel3 import Sentinel3SynergyProcessor
# from .worldcover import ESAWorldCoverProcessor

# __all__ = [
#     'ECOL2TLSTEProcessor',
#     'LandsatC2L2Processor',
#     'Modis13Q1Processor',
#     'OPERASentinel1RTCProcessor',
#     'Sentinel1RTCProcessor',
#     'Sentinel2L2AProcessor',
#     'Sentinel3SynergyProcessor',
#     'ESAWorldCoverProcessor',
# ]

from .base import BaseProcessor, get_default_processor, register_default_processor

__all__ = [
    'BaseProcessor',
    'get_default_processor',
    'register_default_processor',
]

"""
* Creation Date:      2025-10-30
* Provider:           earthaccess
* Collection:         ECO_L2T_LSTE
* Manually modified:  Yes
"""

from datetime import date

from stacathome.metadata import CollectionMetadata, register_static_metadata, Variable

__all__ = [
    'creation_date',
    'provider',
    'collection',
    'metadata',
]

creation_date = date.fromisoformat('2025-10-30')
provider = 'earthaccess'
collection = 'ECO_L2T_LSTE'

metadata = CollectionMetadata(
    Variable(
        name='LST',
        longname='Land Surface Temperature',
        unit='kelvin',
        roles=['data'],
        dtype='float32',
        preferred_resampling='bilinear',
        nodata_value=None,
        offset=None,
        scale=None,
        spatial_resolution=70.0,
    ),
    Variable(
        name='LST_err',
        longname='Land Surface Temperature Error',
        unit='kelvin',
        roles=['data'],
        dtype='float32',
        preferred_resampling='bilinear',
        nodata_value=None,
        offset=None,
        scale=None,
        spatial_resolution=70.0,
    ),
    Variable(
        name='EmisWB',
        longname='Broadband Emissivity',
        unit=None,
        roles=['data'],
        dtype='float32',
        preferred_resampling='bilinear',
        nodata_value=None,
        offset=None,
        scale=None,
        spatial_resolution=70.0,
    ),
    Variable(
        name='view_zenith',
        longname='Zenith angle of the observation',
        unit='degrees',
        roles=['data'],
        dtype='float32',
        preferred_resampling='bilinear',
        nodata_value=None,
        offset=None,
        scale=None,
        spatial_resolution=70.0,
    ),
    Variable(
        name='QC',
        longname='Quality control',
        unit=None,
        roles=['data'],
        dtype='uint16',
        preferred_resampling='nearest',
        nodata_value=0,
        offset=None,
        scale=None,
        spatial_resolution=70.0,
    ),
    Variable(
        name='cloud',
        longname='cloud mask',
        unit=None,
        roles=['data'],
        dtype='uint8',
        preferred_resampling='nearest',
        nodata_value=255,
        offset=None,
        scale=None,
        spatial_resolution=70.0,
    ),
    Variable(
        name='water',
        longname='water mask',
        unit=None,
        roles=['data'],
        dtype='uint8',
        preferred_resampling='nearest',
        nodata_value=255,
        offset=None,
        scale=None,
        spatial_resolution=70.0,
    ),
    Variable(
        name='height',
        longname='SRTM height',
        unit=None,
        roles=['data'],
        dtype='float32',
        preferred_resampling='nearest',
        nodata_value=None,
        offset=None,
        scale=None,
        spatial_resolution=70.0,
    ),
)

register_static_metadata(provider, collection, metadata)

import logging

from shapely import Polygon

from .common import STACItemProcessor


class Sentinel3SynergyProcessor(STACItemProcessor):
    datetime_id = "datetime"
    x = 'longitude'
    y = 'latitude'
    gridded = False
    cubing = 'custom'

    keys = [
        'geolocation'
        'syn-amin',
        'syn-flags',
        'syn-ato550',
        'syn-angstrom-exp550',
        'syn-s1n-reflectance',
        'syn-s1o-reflectance',
        'syn-s2n-reflectance',
        'syn-s2o-reflectance',
        'syn-s3n-reflectance',
        'syn-s3o-reflectance',
        'syn-s5n-reflectance',
        'syn-s5o-reflectance',
        'syn-s6n-reflectance',
        'syn-s6o-reflectance',
        'syn-oa01-reflectance',
        'syn-oa02-reflectance',
        'syn-oa03-reflectance',
        'syn-oa04-reflectance',
        'syn-oa05-reflectance',
        'syn-oa06-reflectance',
        'syn-oa07-reflectance',
        'syn-oa08-reflectance',
        'syn-oa09-reflectance',
        'syn-oa10-reflectance',
        'syn-oa11-reflectance',
        'syn-oa12-reflectance',
        'syn-oa16-reflectance',
        'syn-oa17-reflectance',
        'syn-oa18-reflectance',
        'syn-oa21-reflectance',
    ]

    def get_crs(self):
        return 4326

    @classmethod
    def get_supported_bands(cls):
        return cls.keys

    def get_data_coverage_geometry(self):
        return Polygon(*self.item.geometry["coordinates"])

    def snap_bbox_to_grid(self, bbox):
        # Sentinel-3 data is not in a grid format
        logging.warning("Sentinel-3 data is not in a grid format. No snapping applied.")
        return
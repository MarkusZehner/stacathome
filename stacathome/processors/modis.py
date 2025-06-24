import math

from pyproj import CRS
from shapely import box, Polygon

from .common import Band, STACItemProcessor


class Modis13Q1Processor(STACItemProcessor):
    tilename = "modis:tile-id"
    datetime_id = "start_datetime"
    gridded = True

    indices = ["250m_16_days_NDVI", "250m_16_days_EVI"]

    reflectances = [
        "250m_16_days_MIR_reflectance",
        "250m_16_days_NIR_reflectance",
        "250m_16_days_red_reflectance",
        "250m_16_days_blue_reflectance",
    ]

    sinusoidal_pixel_spacing = 231.65635826

    indices_bands = {}
    for i in indices:
        indices_bands[i] = Band(
            name=i,
            data_type='int16',
            nodata_value=-3000,
            spatial_resolution=sinusoidal_pixel_spacing,
            continuous_measurement=True,
            # kwargs
            scale_factor=0.0001,
            valid_range=(-2000, 10000),
        )

    reflectance_bands = {}
    for i in reflectances:
        reflectance_bands[i] = Band(
            name=i,
            data_type='int16',
            nodata_value=-1000,
            spatial_resolution=sinusoidal_pixel_spacing,
            continuous_measurement=True,
            # kwargs
            scale_factor=0.0001,
            valid_range=(0, 10000),
        )

    other_bands = {
        "250m_16_days_VI_Quality": Band(
            name="250m_16_days_VI_Quality",
            data_type='uint16',
            nodata_value=65535,
            spatial_resolution=sinusoidal_pixel_spacing,
            continuous_measurement=True,
            # kwargs
            valid_range=(0, 65534),
        ),
        "250m_16_days_pixel_reliability": Band(
            name="250m_16_days_pixel_reliability",
            data_type='int16',
            nodata_value=-1,
            spatial_resolution=sinusoidal_pixel_spacing,
            continuous_measurement=False,
            # kwargs
            valid_range=(0, 3),
        ),
        "250m_16_days_relative_azimuth_angle": Band(
            name="250m_16_days_relative_azimuth_angle",
            data_type='int16',
            nodata_value=-4000,
            spatial_resolution=sinusoidal_pixel_spacing,
            continuous_measurement=True,
            # kwargs
            scale_factor=0.01,
            valid_range=(-18000, 18000),
        ),
    }

    special_bands = {}
    all_bands = reflectance_bands | indices_bands | other_bands | special_bands
    supported_bands = list(all_bands.keys())

    def get_crs(self):
        return CRS.from_wkt(self.item.properties['proj:wkt2'])

    def get_bbox(self):
        return Polygon(self.item.properties['proj:geometry']['coordinates'][0])

    def snap_bbox_to_grid(self, bbox):
        # fits modis level 3 sinusoidal grid

        tile_size_m = 1111950.519667
        pixel_size = 231.656358263958
        n_tiles_h = 36
        n_tiles_v = 18
        x_min_global = -tile_size_m * (n_tiles_h / 2)
        y_min_global = -tile_size_m * (n_tiles_v / 2)

        xmin, ymin, xmax, ymax = bbox.bounds

        # Clip bbox to MODIS global bounds
        xmin = max(xmin, x_min_global)
        xmax = min(xmax, x_min_global + tile_size_m * n_tiles_h)
        ymin = max(ymin, y_min_global)
        ymax = min(ymax, y_min_global + tile_size_m * n_tiles_v)

        # Compute tile indices
        x_min_pix = int(math.floor((xmin - x_min_global) / pixel_size) - n_tiles_h / 2 * 4800) * pixel_size
        x_max_pix = int(math.ceil((xmax - x_min_global) / pixel_size) - n_tiles_h / 2 * 4800) * pixel_size
        y_min_pix = int(math.floor((ymin - y_min_global) / pixel_size) - n_tiles_v / 2 * 4800) * pixel_size
        y_max_pix = int(math.ceil((ymax - y_min_global) / pixel_size) - n_tiles_v / 2 * 4800) * pixel_size

        return box(x_min_pix, y_min_pix, x_max_pix, y_max_pix)

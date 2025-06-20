import math

from shapely import box

from .common import STACItemProcessor, Band


class ESAWorldCoverProcessor(STACItemProcessor):
    tilename = "esa_worldcover:product_tile"
    datetime_id = "start_datetime"
    x = 'longitude'
    y = 'latitude'
    gridded = True

    special_bands = {
        "map":
            Band(
                name="map",
                data_type="uint8",
                nodata_value=0,
                spatial_resolution=1 / 12000,
                continuous_measurement=False,
                long_name="ESA WorldCover product 2020/2021",
                flag_meanings=[
                    "Tree cover",
                    "Shrubland",
                    "Grassland",
                    "Cropland",
                    "Built-up",
                    "Bare / sparse vegetation",
                    "Snow and ice",
                    "Permanent water bodies",
                    "Herbaceous wetland",
                    "Mangroves",
                    "Moss and lichen",
                ],
                flag_values=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
                metadata={
                    "color_bar_name": "LC Class",
                    "color_value_max": 100,
                    "color_value_min": 10,
                    "keywords": ["ESA WorldCover", "Classes"],
                },
            ),
        "input_quality":
            Band(
                name="input_quality",
                data_type="int16",
                nodata_value=-1,
                spatial_resolution=1 / 2000,
                continuous_measurement=False,
                long_name="Input Quality Map"
            )
    }

    supported_bands = list(special_bands.keys())
    all_bands = special_bands

    def snap_bbox_to_grid(self, bbox):
        pixel_size = 1 / 12000
        x_min_global = -180
        x_max_global = 180
        y_min_global = -90
        y_max_global = 82.75

        xmin, ymin, xmax, ymax = bbox.bounds

        # Clip bbox to MODIS global bounds
        xmin = max(xmin, x_min_global)
        xmax = min(xmax, x_max_global)
        ymin = max(ymin, y_min_global)
        ymax = min(ymax, y_max_global)

        # Compute tile indices
        x_min_pix = int(math.floor((xmin - x_min_global) / pixel_size)) * pixel_size - 180
        x_max_pix = int(math.ceil((xmax - x_min_global) / pixel_size)) * pixel_size - 180
        y_min_pix = int(math.floor((ymin - y_min_global) / pixel_size)) * pixel_size - 90
        y_max_pix = int(math.ceil((ymax - y_min_global) / pixel_size)) * pixel_size - 90

        return box(x_min_pix, y_min_pix, x_max_pix, y_max_pix)

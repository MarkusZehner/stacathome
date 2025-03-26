import pandas as pd
from pyproj import CRS
from shapely import box, Polygon, transform, distance

from .utils import get_transform


def base_attrs():
    BASE_ATTRS = {
        'Name': 'WeGenS2Dataset',
        'Version': '0.0',
    }
    return BASE_ATTRS


class Band():
    def __init__(self, name, data_type, no_data_value, spatial_resolution, scale_factor, continuous=True, valid_range=None):
        self.name = name
        self.data_type = data_type
        self.no_data_value = no_data_value
        self.spatial_resolution = spatial_resolution
        self.scale_factor = scale_factor
        self.continuous = continuous
        self.valid_range = valid_range

    def __str__(self):
        return f"Band: {self.name}, Data Type: {self.data_type}, NoData Value: {self.no_data_value}, " \
            f"Spatial Resolution: {self.spatial_resolution}, Scale Factor: {self.scale_factor}, " \
            f"Valid Range: {self.valid_range}, Measurement Type: {self.measurement_type}"


class STACCollection():
    def __init__(self, name, filter_arg, grid, attributes):
        self.name = name
        self.filter_arg = filter_arg
        self.grid = grid
        self.attributes = attributes

    def __str__(self):
        return f"Collection: {self.name}, Filter Argument: {self.filter_arg}, Grid: {self.grid}, Attributes: {self.attributes}"


def supported_mspc_collections(S2_tile_grid: str = None, MODIS_grid: str = None, ESA_WorldCover_grid: str = None):
    """
    Returns the supported collections for the MSPC
    """

    return {
        "sentinel-2-l2a": {
            "filter_arg": "s2:mgrs_tile",
            "grid": S2_tile_grid,
            "attributes": sentinel_2_attributes()
        },
        "modis-13Q1-061": {
            "filter_arg": "modis:tile-id",
            "grid": MODIS_grid,
            "attributes": modis_16d_attributes()
        },
        "esa-worldcover": {
            "filter_arg": "esa_worldcover:product_tile",
            "grid": ESA_WorldCover_grid,
            "attributes": esa_wc_attributes(),
        },
        "sentinel-3-synergy-syn-l2-netcdf": {
            "filter_arg": None,
            "grid": None,
            "attributes": None,
        },
    }


def get_stac_filter_arg(collection, collections=None):
    """
    returns the filter argument for the collection
    """
    if collections is None:
        collections = supported_mspc_collections()
    if collection not in collections:
        raise ValueError(f"Collection {collection} not supported")
    return collections[collection]["filter_arg"]


def get_grid(collection, collections=None):
    """
    returns the grid for the collection
    """
    if collections is None:
        collections = supported_mspc_collections()
    if collection not in collections:
        raise ValueError(f"Collection {collection} not supported")
    return collections[collection]["grid"]


def get_attributes(collection, collections=None):
    """
    returns the attributes for the collection
    """
    if collections is None:
        collections = supported_mspc_collections()
    if collection not in collections:
        raise ValueError(f"Collection {collection} not supported")
    return collections[collection]["attributes"]


def sentinel_2_attributes():
    # Define data in a compact format
    s2_bands = [
        ("B01", "uint16", 0, 60, 442.7, 20, 442.3, 20, 129, 129, "continuous"),
        ("B02", "uint16", 0, 10, 492.7, 65, 492.3, 65, 128, 154, "continuous"),
        ("B03", "uint16", 0, 10, 559.8, 35, 558.9, 35, 128, 168, "continuous"),
        ("B04", "uint16", 0, 10, 6064.6, 30, 664.9, 31, 108, 142, "continuous"),
        ("B05", "uint16", 0, 20, 704.1, 14, 703.8, 15, 74.5, 117, "continuous"),
        ("B06", "uint16", 0, 20, 740.5, 14, 739.1, 13, 68, 89, "continuous"),
        ("B07", "uint16", 0, 20, 782.8, 19, 779.7, 19, 67, 105, "continuous"),
        ("B08", "uint16", 0, 10, 832.8, 105, 832.9, 104, 103, 174, "continuous"),
        ("B8A", "uint16", 0, 20, 864.7, 21, 864.0, 21, 52.5, 72, "continuous"),
        ("B09", "uint16", 0, 60, 945.1, 19, 943.2, 20, 9, 114, "continuous"),
        # ("B10", "uint16", 0, 60, 1373.5, 29, 1376.9, 29, 6, 50), not present in MSPC
        ("B11", "uint16", 0, 20, 1613.7, 90, 1610.4, 94, 4, 100, "continuous"),
        ("B12", "uint16", 0, 20, 2202.4, 174, 2185.7, 184, 1.5, 100, "continuous"),
        ("SCL", "uint8", 0, 20, None, None, None, None, None, None, "distinct"),
    ]

    # Create DataFrame
    s2_data_attrs = pd.DataFrame(
        s2_bands,
        columns=[
            "Band",
            "Data Type",
            "NoData Value",
            "Spatial Resolution",
            "S-2A Central Wavelength",
            "S-2A Bandwidth",
            "S-2B Central Wavelength",
            "S-2B Bandwidth",
            "Reference Radiance",
            "SNR at Lref",
            "Measurement Type",
        ],
    )

    sentinel_2_scl_attrs = {
        "long_name": "Scene Classification Layer",
        "units": "n.a.",
        "flag_meanings": [
            "Saturated / Defective",
            "Dark Area Pixels",
            "Cloud Shadows",
            "Vegetation",
            "Bare Soils",
            "Water",
            "Clouds low probability / Unclassified",
            "Clouds medium probability",
            "Clouds high probability",
            "Cirrus",
            "Snow / Ice",
        ],
        "flag_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "DataType": "uint8",
        "NoData": 0,
    }

    return {
        'data_attrs': s2_data_attrs,
        'SCL': sentinel_2_scl_attrs,
    }


def modis_16d_attributes():
    modis_data = [
        ("250m_16_days_NDVI", "int16", -3000, 250, 0.0001, (-2000, 10000), "continuous"),
        ("250m_16_days_EVI", "int16", -3000, 250, 0.0001, (-2000, 10000), "continuous"),
        ("250m_16_days_VI_Quality", "uint16", 65535, 250, None, (0, 65534), "continuous"),
        ("250m_16_days_MIR_reflectance", "int16", -1000, 250, 0.0001, (0, 10000), "continuous"),
        ("250m_16_days_NIR_reflectance", "int16", -1000, 250, 0.0001, (0, 10000), "continuous"),
        ("250m_16_days_red_reflectance", "int16", -1000, 250, 0.0001, (0, 10000), "continuous"),
        ("250m_16_days_blue_reflectance", "int16", -1000, 250, 0.0001, (0, 10000), "continuous"),
        ("250m_16_days_sun_zenith_angle", "int16", -10000, 250, 0.01, (0, 18000), "continuous"),
        ("250m_16_days_pixel_reliability", "int16", -1, 250, None, (0, 3), "distinct"),
        ("250m_16_days_view_zenith_angle", "int16", -10000, 250, 0.01, (0, 18000), "continuous"),
        ("250m_16_days_relative_azimuth_angle", "int16", -4000, 250, 0.01, (-18000, 18000), "continuous"),
    ]
    # TODO: depending on categorical or continuous data, add a 'categorical or continuous' key to the asset_specs

    # Create a DataFrame
    modis_attrs = pd.DataFrame(
        modis_data,
        columns=[
            "Band",
            "Data Type",
            "NoData Value",
            "Spatial Resolution",
            "Scale Factor",
            "Valid Range",
            "Measurement Type",
        ],
    )
    return {"data_attrs": modis_attrs}


def esa_wc_attributes():
    esa_wc_map_attrs = {
        "long_name": "ESA WorldCover product 2021",
        "dims": ["y", "x"],
        "flag_meanings": [
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
        "flag_values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
        "metadata": {
            "color_bar_name": "LC Class",
            "color_value_max": 100,
            "color_value_min": 10,
            "keywords": ["ESA WorldCover", "Classes"],
        },
        "name": "WorldCover21",
        "sources": ["https://planetarycomputer.microsoft.com/api/stac/v1/collections/esa-worldcover"],
        "units": "n.a.",
        "NoData": 0,
        "DataType": "uint8",
        # 'AREA_OR_POINT' :   ,
        # 'add_offset' :  ,
    }
    esa_wc_data = [
        ("map", "uint8", 0, 10, "distinct"),
        ("input_quality", "int16", -1, 60, "distinct"),
    ]

    # Create a DataFrame
    esa_wc_attrs = pd.DataFrame(
        esa_wc_data,
        columns=[
            "Band",
            "Data Type",
            "NoData Value",
            "Spatial Resolution",
            "Measurement Type",
        ],
    )
    return {"esa_worldcover": esa_wc_map_attrs, 'data_attrs': esa_wc_attrs}


def get_band_attributes_s2(attr_dict, cube_vars):
    out_attrs = {}
    for attr_index in attr_dict.index:
        dict_temp = attr_dict.iloc[attr_index].to_dict()
        if dict_temp['Band'] in cube_vars:
            out_attrs[dict_temp['Band']] = {
                "long_name": f"Reflectance in band {dict_temp['Band']}",
                "source": "https://planetarycomputer.microsoft.com/api/stac/v1/sentinel-2-l2a",
                "units": ("n.a.",),
                "NoData": "nan",
                "metadata": {
                    "color_bar_name": "gray",
                    "color_value_max": 1.0,
                    "color_value_min": 0.0,
                    "keywords": ["Sentinel-2", "Reflectances"],
                },
                "centralwavelength": {
                    "S-2A": dict_temp["S-2A Central Wavelength"],
                    "S-2B": dict_temp["S-2B Central Wavelength"],
                },
                "bandwidth": {
                    "S-2A": dict_temp["S-2A Bandwidth"],
                    "S-2B": dict_temp["S-2B Bandwidth"],
                    "S-2C": 'to be added',
                },
                "Data Type": "float32",
            }
    return out_attrs


def get_band_attributes_modis(attr_dict, cube_vars):
    out_attrs = {}
    for attr_index in attr_dict.index:
        dict_temp = attr_dict.iloc[attr_index].to_dict()
        if dict_temp['Band'] in cube_vars:
            out_attrs[dict_temp['Band']] = {
                "long_name": f"16 Day Reflectance in band {dict_temp['Band']}",
                "source": "https://planetarycomputer.microsoft.com/api/stac/v1/modis-13Q1-061",
                "units": ("n.a.",),
                "NoData": dict_temp['NoData Value'],
                "metadata": {
                    "color_bar_name": "gray",
                    "color_value_max": 1.0,
                    "color_value_min": 0.0,
                    "keywords": ["MODIS", "Reflectances", "Indices"],
                },
                "valid range": dict_temp['Valid Range'],
                "Scale Factor": dict_temp['Scale Factor'],
                "Data Type": dict_temp['Data Type'],
            }
    return out_attrs


def add_attributes(cube):
    for collection in supported_mspc_collections().keys():
        attributes = get_attributes(collection)
        data_atts = attributes.pop("data_attrs")

        if data_atts is not None:
            if collection == 'sentinel-2-l2a':
                band_attrs = get_band_attributes_s2(data_atts, cube.data_vars.keys())
            elif collection == 'modis-13Q1-061':
                band_attrs = get_band_attributes_modis(data_atts, cube.data_vars.keys())
            else:
                band_attrs = {}
            for band in band_attrs.keys():
                cube[band].attrs = band_attrs[band]
        if attributes is not None:
            for key in attributes.keys():
                match = next((item for item in cube.data_vars.keys() if item == key or item.startswith(key)), None)
                if match is not None:
                    cube[match].attrs = attributes[key]
    return cube


def get_resampling_per_band(target_res, bands, collection):
    attributes = get_attributes(collection)['data_attrs']
    band_names = list(attributes['Band'])
    resolutions = list(attributes['Spatial Resolution'])
    measurement_type = list(attributes['Measurement Type'])

    resampling_scheme = {}
    for band, res, mtype in zip(band_names, resolutions, measurement_type):
        if mtype == "continuous":
            resampling_scheme[band] = "nearest" if target_res <= res else "bilinear"
        else:
            resampling_scheme[band] = "nearest" if target_res <= res else "mode"

    return {k: v for k, v in resampling_scheme.items() if k in bands}


def transform_asset_bbox(asset, to=4326, collection='sentinel-2-l2a'):
    if collection == 'sentinel-2-l2a':
        keys = list(get_attributes('sentinel-2-l2a')['data_attrs']['Band'])
        k = next((key for key in keys if key in asset.assets), None)
        bbox = box(*asset.assets[k].extra_fields['proj:bbox'])
        tr = get_transform(asset.properties['proj:epsg'], to)
    if collection == 'modis-13Q1-061':
        # keys = list(get_attributes('modis-13Q1-061')['data_attrs']['Band'])
        # k = next((key for key in keys if key in asset.assets), None)
        bbox = Polygon(asset.properties['proj:geometry']['coordinates'][0])
        tr = get_transform(CRS.from_wkt(asset.properties['proj:wkt2']), to)
    if collection == 'esa-worldcover':
        bbox = box(*asset.bbox)
        tr = get_transform(asset.properties['proj:epsg'], to)
    return transform(bbox, tr)


def get_asset_crs(asset, collection='sentinel-2-l2a'):
    if collection == 'sentinel-2-l2a':
        return asset.properties['proj:epsg']
    if collection == 'modis-13Q1-061':
        return CRS.from_wkt(asset.properties['proj:wkt2'])
    if collection == 'esa-worldcover':
        return asset.properties['proj:epsg']


def get_asset_box_and_transform(asset, collection='sentinel-2-l2a', from_crs=4326):
    to_crs = get_asset_crs(asset, collection)
    if collection == 'sentinel-2-l2a':
        keys = list(get_attributes('sentinel-2-l2a')['data_attrs']['Band'])
        k = next((key for key in keys if key in asset.assets), None)
        bbox = box(*asset.assets[k].extra_fields['proj:bbox'])
        tr = get_transform(from_crs, to_crs)
    if collection == 'modis-13Q1-061':
        bbox = Polygon(asset.properties['proj:geometry']['coordinates'][0])
        tr = get_transform(from_crs, to_crs)
    if collection == 'esa-worldcover':
        bbox = box(*asset.bbox)
        tr = get_transform(from_crs, to_crs)
    return bbox, tr


def contains_in_native_crs(asset, compare_box, compare_box_crs=4326, collection='sentinel-2-l2a'):
    asset_bbox, tr = get_asset_box_and_transform(asset=asset, collection=collection, from_crs=compare_box_crs)
    bbox = transform(compare_box, tr)
    return asset_bbox.contains(bbox)


def centroid_distance_in_native_crs(asset, compare_box, compare_box_crs=4326, collection='sentinel-2-l2a'):
    asset_bbox, tr = get_asset_box_and_transform(asset=asset, collection=collection, from_crs=compare_box_crs)
    bbox = transform(compare_box, tr)
    return distance(asset_bbox.centroid, bbox.centroid)


def intersection_area_percent_in_native_crs(asset, compare_box, compare_box_crs=4326, collection='sentinel-2-l2a'):
    asset_bbox, tr = get_asset_box_and_transform(asset=asset, collection=collection, from_crs=compare_box_crs)
    bbox = transform(compare_box, tr)
    return asset_bbox.intersection(bbox).area / bbox.area

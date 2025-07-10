import pystac
from odc.geo import AnchorEnum
from odc.geo.geobox import GeoBox
from stacathome.stac import geobox_from_asset, geoboxes_from_assets


TEST_ITEM_DICT = {
    'type': 'Feature',
    'stac_version': '1.1.0',
    'stac_extensions': [
        'https://stac-extensions.github.io/eo/v1.1.0/schema.json',
        'https://stac-extensions.github.io/sat/v1.0.0/schema.json',
        'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
    ],
    'id': 'S2B_MSIL2A_20230101T102339_R065_T32UPC_20240806T223544',
    'geometry': {
        'type': 'Polygon',
        'coordinates': [
            [
                [10.4678842, 52.3413551],
                [12.0777579, 52.3103669],
                [12.0112541, 51.3245234],
                [10.4361206, 51.3544394],
                [10.4678842, 52.3413551],
            ]
        ],
    },
    'bbox': [10.4361206, 51.3245234, 12.0777579, 52.3413551],
    'properties': {
        'datetime': '2023-01-01T10:23:39.024000Z',
        'platform': 'Sentinel-2B',
        'instruments': ['msi'],
        's2:mgrs_tile': '32UPC',
        'constellation': 'Sentinel 2',
        'eo:cloud_cover': 99.999923,
        'sat:orbit_state': 'descending',
        'sat:relative_orbit': 65,
        'proj:code': 'EPSG:32632',
    },
    'links': [
        {
            'rel': 'collection',
            'href': 'https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a',
            'type': 'application/json',
        },
        {
            'rel': 'parent',
            'href': 'https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a',
            'type': 'application/json',
        },
        {
            'rel': 'root',
            'href': 'https://planetarycomputer.microsoft.com/api/stac/v1',
            'type': 'application/json',
            'title': 'Microsoft Planetary Computer STAC API',
        },
        {
            'rel': 'self',
            'href': 'https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20230101T102339_R065_T32UPC_20240806T223544',
            'type': 'application/geo+json',
        },
        {'rel': 'license', 'href': 'https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice'},
        {
            'rel': 'preview',
            'href': 'https://planetarycomputer.microsoft.com/api/data/v1/item/map?collection=sentinel-2-l2a&item=S2B_MSIL2A_20230101T102339_R065_T32UPC_20240806T223544',
            'type': 'text/html',
            'title': 'Map of item',
        },
    ],
    'assets': {
        'B01': {
            'href': 'https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/32/U/PC/2023/01/01/S2B_MSIL2A_20230101T102339_N0510_R065_T32UPC_20240806T223544.SAFE/GRANULE/L2A_T32UPC_A030407_20230101T102333/IMG_DATA/R60m/T32UPC_20230101T102339_B01_60m.tif',
            'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
            'title': 'Band 1 - Coastal aerosol - 60m',
            'proj:bbox': [600000.0, 5690220.0, 709800.0, 5800020.0],
            'proj:shape': [1830, 1830],
            'proj:transform': [60.0, 0.0, 600000.0, 0.0, -60.0, 5800020.0],
            'gsd': 60.0,
            'eo:bands': [
                {
                    'name': 'B01',
                    'common_name': 'coastal',
                    'description': 'Band 1 - Coastal aerosol',
                    'center_wavelength': 0.443,
                    'full_width_half_max': 0.027,
                }
            ],
            'roles': ['data'],
        },
        'B09': {
            'href': 'https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/32/U/PC/2023/01/01/S2B_MSIL2A_20230101T102339_N0510_R065_T32UPC_20240806T223544.SAFE/GRANULE/L2A_T32UPC_A030407_20230101T102333/IMG_DATA/R60m/T32UPC_20230101T102339_B09_60m.tif',
            'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
            'title': 'Band 9 - Water vapor - 60m',
            'proj:bbox': [600000.0, 5690220.0, 709800.0, 5800020.0],
            'proj:shape': [1830, 1830],
            'proj:transform': [60.0, 0.0, 600000.0, 0.0, -60.0, 5800020.0],
            'gsd': 60.0,
            'eo:bands': [
                {
                    'name': 'B09',
                    'description': 'Band 9 - Water vapor',
                    'center_wavelength': 0.945,
                    'full_width_half_max': 0.026,
                }
            ],
            'roles': ['data'],
        },
        'B11': {
            'href': 'https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/32/U/PC/2023/01/01/S2B_MSIL2A_20230101T102339_N0510_R065_T32UPC_20240806T223544.SAFE/GRANULE/L2A_T32UPC_A030407_20230101T102333/IMG_DATA/R20m/T32UPC_20230101T102339_B11_20m.tif',
            'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
            'title': 'Band 11 - SWIR (1.6) - 20m',
            'proj:bbox': [600000.0, 5690220.0, 709800.0, 5800020.0],
            'proj:shape': [5490, 5490],
            'proj:transform': [20.0, 0.0, 600000.0, 0.0, -20.0, 5800020.0],
            'gsd': 20.0,
            'eo:bands': [
                {
                    'name': 'B11',
                    'common_name': 'swir16',
                    'description': 'Band 11 - SWIR (1.6)',
                    'center_wavelength': 1.61,
                    'full_width_half_max': 0.143,
                }
            ],
            'roles': ['data'],
        },
        'B12': {
            'href': 'https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/32/U/PC/2023/01/01/S2B_MSIL2A_20230101T102339_N0510_R065_T32UPC_20240806T223544.SAFE/GRANULE/L2A_T32UPC_A030407_20230101T102333/IMG_DATA/R20m/T32UPC_20230101T102339_B12_20m.tif',
            'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
            'title': 'Band 12 - SWIR (2.2) - 20m',
            'proj:bbox': [600000.0, 5690220.0, 709800.0, 5800020.0],
            'proj:shape': [5490, 5490],
            'proj:transform': [20.0, 0.0, 600000.0, 0.0, -20.0, 5800020.0],
            'gsd': 20.0,
            'eo:bands': [
                {
                    'name': 'B12',
                    'common_name': 'swir22',
                    'description': 'Band 12 - SWIR (2.2)',
                    'center_wavelength': 2.19,
                    'full_width_half_max': 0.242,
                }
            ],
            'roles': ['data'],
        },
        'SCL': {
            'href': 'https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/32/U/PC/2023/01/01/S2B_MSIL2A_20230101T102339_N0510_R065_T32UPC_20240806T223544.SAFE/GRANULE/L2A_T32UPC_A030407_20230101T102333/IMG_DATA/R20m/T32UPC_20230101T102339_SCL_20m.tif',
            'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
            'title': 'Scene classfication map (SCL)',
            'proj:bbox': [600000.0, 5690220.0, 709800.0, 5800020.0],
            'proj:shape': [5490, 5490],
            'proj:transform': [20.0, 0.0, 600000.0, 0.0, -20.0, 5800020.0],
            'gsd': 20.0,
            'roles': ['data'],
        },
        'WVP': {
            'href': 'https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/32/U/PC/2023/01/01/S2B_MSIL2A_20230101T102339_N0510_R065_T32UPC_20240806T223544.SAFE/GRANULE/L2A_T32UPC_A030407_20230101T102333/IMG_DATA/R10m/T32UPC_20230101T102339_WVP_10m.tif',
            'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
            'title': 'Water vapour (WVP)',
            'proj:bbox': [600000.0, 5690220.0, 709800.0, 5800020.0],
            'proj:shape': [10980, 10980],
            'proj:transform': [10.0, 0.0, 600000.0, 0.0, -10.0, 5800020.0],
            'gsd': 10.0,
            'roles': ['data'],
        },
        'NO_BOX_TEST': {
            'href': 'https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/32/U/PC/2023/01/01/S2B_MSIL2A_20230101T102339_N0510_R065_T32UPC_20240806T223544.SAFE/GRANULE/L2A_T32UPC_A030407_20230101T102333/IMG_DATA/R10m/T32UPC_20230101T102339_WVP_10m.ti',
            'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
            'title': 'No GeoBox',
            'roles': ['data'],
            'proj:transform': [
                10.0,
                0.0,
                600000.0,
                0.0,
                -10.0,
                5800020.0,
            ],  # only partial info, not enough to construct GeoBox
            'gsd': 10.0,
        },
    },
    'collection': 'sentinel-2-l2a',
}


def _get_test_item() -> pystac.Item:
    return pystac.Item.from_dict(TEST_ITEM_DICT)


class TestGeoboxesFromAssets:

    def test_asset_ids_param(self):
        # Accepts single string
        geoboxes = geoboxes_from_assets(_get_test_item(), 'B01')
        assert geoboxes is not None
        assert isinstance(geoboxes, dict)
        assert set(geoboxes) == {'B01'}

        # Accepts Iterable
        geoboxes = geoboxes_from_assets(_get_test_item(), ['B01', 'WVP'])
        assert geoboxes is not None
        assert isinstance(geoboxes, dict)
        assert set(geoboxes) == {'B01', 'WVP'}

        # Accepts None and returns all assets with geobox (should not include NO_BOX_TEST)
        geoboxes = geoboxes_from_assets(_get_test_item())
        assert geoboxes is not None
        assert isinstance(geoboxes, dict)
        assert set(geoboxes) == {'B01', 'B09', 'B11', 'B12', 'SCL', 'WVP'}

    def test_correct_box(self):
        geoboxes = geoboxes_from_assets(_get_test_item(), ['B01', 'SCL'])
        b01 = geoboxes['B01']
        scl = geoboxes['SCL']

        assert isinstance(b01, GeoBox)
        assert b01.resolution.xy == (60, -60)
        assert b01.crs is not None
        assert b01.crs.epsg == 32632
        assert b01.shape == (1830, 1830)
        assert b01.pix2wld(0, 0) == (600000.0, 5800020.0)
        assert b01.pix2wld(1, 0) == (600060.0, 5800020.0)

        assert isinstance(scl, GeoBox)
        assert scl.resolution.xy == (20, -20)
        assert scl.crs is not None
        assert scl.crs.epsg == 32632
        assert scl.shape == (5490, 5490)
        assert scl.pix2wld(0, 0) == (600000.0, 5800020.0)
        assert scl.pix2wld(1, 0) == (600020.0, 5800020.0)

        # Matching bbox in local CRS
        assert b01.boundingbox == scl.boundingbox

        # Anchors
        assert b01.anchor == AnchorEnum.EDGE

    def test_no_entries_for_partial_proj(self):
        geoboxes = geoboxes_from_assets(_get_test_item(), 'NO_BOX_TEST')
        assert len(geoboxes) == 0

    def test_no_entries_for_nonexisting_assets(self):
        geoboxes = geoboxes_from_assets(_get_test_item(), ['ABCD', 'XYZ'])
        assert len(geoboxes) == 0


class TestGeoboxFromAsset:

    def test_existing_asset(self):
        b01 = geobox_from_asset(_get_test_item(), 'B01')
        assert b01 is not None
        assert isinstance(b01, GeoBox)
        assert b01.resolution.xy == (60, -60)
        assert b01.crs is not None
        assert b01.crs.epsg == 32632
        assert b01.shape == (1830, 1830)

    def test_partial_asset(self):
        assert geobox_from_asset(_get_test_item(), 'NO_BOX_TEST') is None

    def test_nonsense_asset(self):
        assert geobox_from_asset(_get_test_item(), 'ABCD') is None

import odc.geo.geom as geom
import pystac
import shapely
from odc.geo import AnchorEnum
from odc.geo.geobox import Affine, CRS, GeoBox
from stacathome.stac import enclosing_geoboxes_per_grid, EnclosingGeoboxResult, geobox_from_asset, geoboxes_from_assets


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

# Grids / geoboxes contained in the item
TRUE_10M_BOX = GeoBox((10980, 10980), Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 5800020.0), CRS('EPSG:32632'))
TRUE_20M_BOX = GeoBox((5490, 5490), Affine(20.0, 0.0, 600000.0, 0.0, -20.0, 5800020.0), CRS('EPSG:32632'))
TRUE_60M_BOX = GeoBox((1830, 1830), Affine(60.0, 0.0, 600000.0, 0.0, -60.0, 5800020.0), CRS('EPSG:32632'))


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


class TestEnclosingGeoboxesPerGrid:

    def test_type_and_sorting(self):
        geoboxes = enclosing_geoboxes_per_grid(_get_test_item(), TRUE_60M_BOX.extent)
        assert isinstance(geoboxes, list)
        assert len(geoboxes) == 3
        assert isinstance(geoboxes[0], EnclosingGeoboxResult)
        assert [g.gsd for g in geoboxes] == [10.0, 20.0, 60.0]

    def test_identical_geometry(self):
        # This is the most simplistic case:
        # We pass the original shape of the GeoBox contained in the item (a rectangle in the local CRS),
        # and expect to retrieve identical geoboxes to those resolutions contained in the item.
        geoboxes = enclosing_geoboxes_per_grid(_get_test_item(), TRUE_60M_BOX.extent)

        assert geoboxes[0].grid_box == TRUE_10M_BOX
        assert geoboxes[0].enclosing_box == TRUE_10M_BOX

        assert geoboxes[1].grid_box == TRUE_20M_BOX
        assert geoboxes[1].enclosing_box == TRUE_20M_BOX

        assert geoboxes[2].grid_box == TRUE_60M_BOX
        assert geoboxes[2].enclosing_box == TRUE_60M_BOX

    def test_point_geometry(self):
        # For a single point we expect to get three geoboxes, each describing the pixel containing the point
        # We test the (1,1) pixel for 10m, (0,0) for 20m and 60m
        point = geom.Geometry(shapely.Point(600000.0 + 10.1, 5800020.0 - 10.1), CRS('EPSG:32632'))
        geoboxes = enclosing_geoboxes_per_grid(_get_test_item(), point)

        box_10m_expected = GeoBox(
            shape=(1, 1),
            affine=Affine(10.0, 0.0, 600000.0 + 10.0, 0.0, -10.0, 5800020.0 - 10.0),
            crs=CRS('EPSG:32632'),
        )
        assert geoboxes[0].grid_box == TRUE_10M_BOX
        assert geoboxes[0].enclosing_box == box_10m_expected

        box_20m_expected = GeoBox(
            shape=(1, 1),
            affine=Affine(20.0, 0.0, 600000.0, 0.0, -20.0, 5800020.0),
            crs=CRS('EPSG:32632'),
        )
        assert geoboxes[1].grid_box == TRUE_20M_BOX
        assert geoboxes[1].enclosing_box == box_20m_expected

        box_60m_expected = GeoBox(
            shape=(1, 1),
            affine=Affine(60.0, 0.0, 600000.0, 0.0, -60.0, 5800020.0),
            crs=CRS('EPSG:32632'),
        )
        assert geoboxes[2].grid_box == TRUE_60M_BOX
        assert geoboxes[2].enclosing_box == box_60m_expected

    def test_snapping(self):
        # Here we use a slightly smaller extent whose points additionally lie within pixel boundaries
        # We expect it to be extented to next biggest grid multiple
        geometry = TRUE_60M_BOX.extent.buffer(-25.0)  # -25meters, pixel 2 for 10m,  pixel 1 for 20m,  pixel 0 for 60m,
        geoboxes = enclosing_geoboxes_per_grid(_get_test_item(), geometry)

        box_10m_expected = GeoBox(
            (10980 - 4, 10980 - 4),
            Affine(10.0, 0.0, 600000.0 + 2 * 10, 0.0, -10.0, 5800020.0 - 2 * 10),
            CRS('EPSG:32632'),
        )
        assert geoboxes[0].grid_box == TRUE_10M_BOX
        assert geoboxes[0].enclosing_box == box_10m_expected

        box_20m_expected = GeoBox(
            (5490 - 2, 5490 - 2),
            Affine(20.0, 0.0, 600000.0 + 1 * 20, 0.0, -20.0, 5800020.0 - 1 * 20),
            CRS('EPSG:32632'),
        )
        assert geoboxes[1].grid_box == TRUE_20M_BOX
        assert geoboxes[1].enclosing_box == box_20m_expected

        assert geoboxes[2].grid_box == TRUE_60M_BOX
        assert geoboxes[2].enclosing_box == TRUE_60M_BOX

    def test_outside_box(self):
        # Here we provide coordinates that our completely outside of the original geoboxes
        geometry = TRUE_60M_BOX.left.extent
        geoboxes = enclosing_geoboxes_per_grid(_get_test_item(), geometry)

        assert geoboxes[0].grid_box == TRUE_10M_BOX
        assert geoboxes[0].enclosing_box == TRUE_10M_BOX.left

        assert geoboxes[1].grid_box == TRUE_20M_BOX
        assert geoboxes[1].enclosing_box == TRUE_20M_BOX.left

        assert geoboxes[2].grid_box == TRUE_60M_BOX
        assert geoboxes[2].enclosing_box == TRUE_60M_BOX.left

    def test_complex_geometry(self):
        # We use a skewed quadrangle that reaches over the boundaries of the orinal box
        # Additionally the quadrangle is passed in as WGS84

        def pix2wld(x, y):
            return TRUE_10M_BOX.pix2wld(x, y)

        # bounds: left -3, right: 20, top: 3, bottom: 20
        geometry = geom.polygon(
            [
                pix2wld(3, 3),
                pix2wld(-3, 10),
                pix2wld(0, 15),
                pix2wld(20, 20),
            ],
            CRS('EPSG:32632'),
        )
        geometry = geometry.transform(lambda x, y: (x + 0.1, y + 0.1))
        geometry = geometry.to_crs(CRS('EPSG:4326'), resolution=2)
        geoboxes = enclosing_geoboxes_per_grid(_get_test_item(), geometry)

        box_10m_expected = GeoBox(
            (17, 23),  # y,x !!!
            Affine(10.0, 0.0, 600000.0 - 3 * 10.0, 0.0, -10.0, 5800020.0 - 3 * 10.0),
            CRS('EPSG:32632'),
        )
        assert geoboxes[0].grid_box == TRUE_10M_BOX
        assert geoboxes[0].enclosing_box == box_10m_expected

        box_20m_expected = GeoBox(
            (9, 12),  # y,x !!!
            Affine(20.0, 0.0, 600000.0 - 2 * 20.0, 0.0, -20.0, 5800020.0 - 1 * 20.0),
            CRS('EPSG:32632'),
        )
        assert geoboxes[1].grid_box == TRUE_20M_BOX
        assert geoboxes[1].enclosing_box == box_20m_expected

        box_60m_expected = GeoBox(
            (4, 5),  # y,x !!!
            Affine(60.0, 0.0, 600000.0 - 1 * 60.0, 0.0, -60.0, 5800020.0),
            CRS('EPSG:32632'),
        )
        assert geoboxes[2].grid_box == TRUE_60M_BOX
        assert geoboxes[2].enclosing_box == box_60m_expected

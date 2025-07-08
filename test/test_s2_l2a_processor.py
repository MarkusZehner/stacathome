import shapely
from odc.geo.geobox import GeoBox

from stacathome.processors.sentinel2_rewrite import Sentinel2L2AProcessor
from stacathome.providers import get_provider


NEAR_TILE_CORNER = shapely.Point(704800, 5895040) # EPSG:32632
NEAR_TILE_CORNER_SHIFT = shapely.Point(710800, 5901040) # EPSG:32632

def test_geobox(center_p, crs='EPSG:32632', resolution=10, size_box=500):
    """Create a geobox centered at the given point with a specified resolution and size."""
    bbox = (
        center_p.x - size_box,  # left
        center_p.y - size_box,  # bottom
        center_p.x + size_box,  # right
        center_p.y + size_box,  # top
    )  # 1km x 1km
    geobox = GeoBox.from_bbox(
        bbox=bbox,
        crs=crs,
        resolution=resolution,
        tight=True,
    )
    return geobox

class TestSentinel2L2AProcessor:

    def test_filtering(self):
        provider = get_provider('planetary_computer')
        processor = Sentinel2L2AProcessor()

        # these contain 6 tiles * 3 time steps * 2 processing times
        expected_ids_raw = [
            'S2A_MSIL2A_20230106T102411_R065_T33UUV_20240807T164608',
            'S2A_MSIL2A_20230106T102411_R065_T33UUV_20230106T233729',
            'S2A_MSIL2A_20230106T102411_R065_T33UUU_20240807T164608',
            'S2A_MSIL2A_20230106T102411_R065_T33UUU_20230106T233541',
            'S2A_MSIL2A_20230106T102411_R065_T32UQE_20240807T164608',
            'S2A_MSIL2A_20230106T102411_R065_T32UQE_20230107T001414',
            'S2A_MSIL2A_20230106T102411_R065_T32UQD_20240807T164608',
            'S2A_MSIL2A_20230106T102411_R065_T32UQD_20230106T235342',
            'S2A_MSIL2A_20230106T102411_R065_T32UPE_20240807T164608',
            'S2A_MSIL2A_20230106T102411_R065_T32UPE_20230107T001259',
            'S2A_MSIL2A_20230106T102411_R065_T32UPD_20240807T164608',
            'S2A_MSIL2A_20230106T102411_R065_T32UPD_20230107T001750',
            'S2B_MSIL2A_20230104T103329_R108_T33UUV_20240809T082154',
            'S2B_MSIL2A_20230104T103329_R108_T33UUV_20230104T214938',
            'S2B_MSIL2A_20230104T103329_R108_T33UUU_20240809T082154',
            'S2B_MSIL2A_20230104T103329_R108_T33UUU_20230104T203730',
            'S2B_MSIL2A_20230104T103329_R108_T32UQE_20240809T082154',
            'S2B_MSIL2A_20230104T103329_R108_T32UQE_20230104T215054',
            'S2B_MSIL2A_20230104T103329_R108_T32UQD_20240809T082154',
            'S2B_MSIL2A_20230104T103329_R108_T32UQD_20230104T192325',
            'S2B_MSIL2A_20230104T103329_R108_T32UPE_20240809T082154',
            'S2B_MSIL2A_20230104T103329_R108_T32UPE_20230104T192625',
            'S2B_MSIL2A_20230104T103329_R108_T32UPD_20240809T082154',
            'S2B_MSIL2A_20230104T103329_R108_T32UPD_20230104T200418',
            'S2B_MSIL2A_20230101T102339_R065_T33UUV_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T33UUV_20230101T222810',
            'S2B_MSIL2A_20230101T102339_R065_T33UUU_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T33UUU_20230101T221642',
            'S2B_MSIL2A_20230101T102339_R065_T32UQE_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T32UQE_20230101T231800',
            'S2B_MSIL2A_20230101T102339_R065_T32UQD_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T32UQD_20230101T231951',
            'S2B_MSIL2A_20230101T102339_R065_T32UPE_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T32UPE_20230101T225700',
            'S2B_MSIL2A_20230101T102339_R065_T32UPD_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T32UPD_20230101T231205',
        ]


        filter_1 = [
            'S2A_MSIL2A_20230106T102411_R065_T32UQD_20240807T164608',
            'S2B_MSIL2A_20230104T103329_R108_T32UQD_20240809T082154',
            'S2B_MSIL2A_20230101T102339_R065_T32UQD_20240806T223544',
            ]
        
        filter_2 = [
            'S2A_MSIL2A_20230106T102411_R065_T33UUU_20240807T164608',
            'S2B_MSIL2A_20230104T103329_R108_T33UUU_20240809T082154',
            'S2B_MSIL2A_20230101T102339_R065_T33UUU_20240806T223544',
            ]

        geobox = test_geobox(NEAR_TILE_CORNER, resolution=10, size_box=5000)
        area_of_interest = geobox.footprint('EPSG:4326', buffer=10, npoints=4)
        items = provider.request_items(
            collection='sentinel-2-l2a',
            starttime='2023-01-01',
            endtime='2023-01-09',
            area_of_interest=area_of_interest,
        )

        filtered_items = processor.filter_items(
            provider=None,
            area_of_interest=geobox.footprint('EPSG:4326', buffer=0, npoints=4),
            items=items,
        )

        geobox_shift = test_geobox(NEAR_TILE_CORNER_SHIFT, resolution=10)
        filtered_items_shift = processor.filter_items(
            provider=None,
            area_of_interest=geobox_shift.footprint('EPSG:4326', buffer=0, npoints=4),
            items=items,
        )

        geobox_large = test_geobox(NEAR_TILE_CORNER, resolution=10, size_box=10000)
        filtered_items_large = processor.filter_items(
            provider=None,
            area_of_interest=geobox_large.footprint('EPSG:4326', buffer=0, npoints=4),
            items=items,
        )

        assert set(item.id for item in items) == set(expected_ids_raw)
        assert set(item.id for item in filtered_items) == set(filter_1)
        assert set(item.id for item in filtered_items_shift) == set(filter_2)

        print([i for i in filtered_items_large])
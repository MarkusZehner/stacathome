import pystac
import pytest
import shapely
from stacathome.metadata import CollectionMetadata
from stacathome.providers import STACProvider


class TestSTACProvider:

    @pytest.mark.remote
    def test_get_items_planetary(self):
        import planetary_computer

        start = '2023-01-01'
        end = '2023-01-02'
        area_of_interest = shapely.box(11, 51, 11.5, 51.5)

        EXPECTED_ITEM_IDS = {
            'S2B_MSIL2A_20230101T102339_R065_T32UPC_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T32UPC_20230101T211104',
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20230101T222806',
        }

        provider = STACProvider(
            url='https://planetarycomputer.microsoft.com/api/stac/v1',
            sign=planetary_computer.sign,
        )

        item_col = provider.request_items(
            collection='sentinel-2-l2a',
            starttime=start,
            endtime=end,
            area_of_interest=area_of_interest,
        )
        assert isinstance(item_col, pystac.ItemCollection)
        assert len(item_col) == 4
        assert {item.id for item in item_col} == EXPECTED_ITEM_IDS

    @pytest.mark.remote
    def test_get_metadata(self):
        import planetary_computer

        provider = STACProvider(
            url='https://planetarycomputer.microsoft.com/api/stac/v1',
            sign=planetary_computer.sign,
        )

        metadata = provider.get_metadata('sentinel-2-l2a')
        assert isinstance(metadata, CollectionMetadata)

        EXPECTED_VARS = {
            'AOT',
            'B01',
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
            'B08',
            'B8A',
            'B09',
            'B11',
            'B12',
            'SCL',
            'WVP',
            'visual',
        }
        assert set(metadata.available_variables()) == EXPECTED_VARS

        b01 = metadata.get_variable('B01')
        assert b01.spatial_resolution == 60.0
        assert b01.center_wavelength == 0.443
        assert b01.full_width_half_max == 0.027
        assert b01.roles == ['data']

    @pytest.mark.remote
    def test_get_metadata_landsat(self):
        import planetary_computer

        provider = STACProvider(
            url='https://planetarycomputer.microsoft.com/api/stac/v1',
            sign=planetary_computer.sign,
        )

        metadata = provider.get_metadata('landsat-c2-l2')
        assert isinstance(metadata, CollectionMetadata)

        print(metadata)

        EXPECTED_VARS = {
            'qa',
            'red',
            'blue',
            'drad',
            'emis',
            'emsd',
            'lwir',
            'trad',
            'urad',
            'atran',
            'cdist',
            'green',
            'nir08',
            'lwir11',
            'swir16',
            'swir22',
            'coastal',
            'atmos_opacity',
        }
        vars = set(metadata.available_variables())
        assert (
            vars == EXPECTED_VARS
        ), f'Missing variables: {EXPECTED_VARS - vars} | Extra variables:{vars - EXPECTED_VARS}'

        green = metadata.get_variable('green')
        assert green.spatial_resolution == 30.0
        assert green.center_wavelength == 0.56
        assert green.scale == 2.75e-05
        assert green.offset == -0.2
        assert green.nodata_value == 0
        assert green.dtype == 'uint16'

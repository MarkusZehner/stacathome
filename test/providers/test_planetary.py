import time
from datetime import date

import planetary_computer
import pystac
import pytest
import shapely
from stacathome.metadata import CollectionMetadata
from stacathome.providers import STACProvider


def construct_provider():
    provider = STACProvider(
        'test_provider',
        url='https://planetarycomputer.microsoft.com/api/stac/v1',
        sign=planetary_computer.sign,
    )
    return provider


class TestPlanetaryComputerProvider:

    @pytest.mark.remote
    @pytest.mark.planetary
    def test_get_item(self):
        provider = construct_provider()
        item = provider.get_item('sentinel-2-l2a', 'S2B_MSIL2A_20230101T102339_R065_T32UPC_20240806T223544')
        assert item is not None
        assert isinstance(item, pystac.Item)
        assert item.collection_id == 'sentinel-2-l2a'
        assert item.id == 'S2B_MSIL2A_20230101T102339_R065_T32UPC_20240806T223544'
        assert item.datetime.date() == date(2023, 1, 1)
        assert 'B01' in item.assets

    @pytest.mark.remote
    @pytest.mark.planetary
    def test_get_item_raises_keyerror(self):
        provider = construct_provider()
        with pytest.raises(KeyError):
            provider.get_item('UNK_COLLECTION', 'S2B_MSIL2A_20230101T102339_R065_T32UPC_20240806T223544')

    @pytest.mark.remote
    @pytest.mark.planetary
    def test_get_item_returns_none(self):
        provider = construct_provider()
        item = provider.get_item('sentinel-2-l2a', 'S2B_UNK_ID')
        assert item is None

    @pytest.mark.remote
    @pytest.mark.planetary
    def test_get_items_planetary(self):
        start = '2023-01-01'
        end = '2023-01-02'
        roi = shapely.box(11, 51, 11.5, 51.5)

        EXPECTED_ITEM_IDS = {
            'S2B_MSIL2A_20230101T102339_R065_T32UPC_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T32UPC_20230101T211104',
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20230101T222806',
        }

        provider = construct_provider()

        item_col = provider.request_items(
            collection='sentinel-2-l2a',
            starttime=start,
            endtime=end,
            roi=roi,
        )
        assert isinstance(item_col, pystac.ItemCollection)
        assert len(item_col) == 4
        assert {item.id for item in item_col} == EXPECTED_ITEM_IDS

    @pytest.mark.remote
    @pytest.mark.planetary
    def test_available_collections(self):
        provider = construct_provider()
        collections = provider.available_collections()
        assert isinstance(collections, list)
        assert len(collections) > 0
        assert isinstance(collections[0], str)

    @pytest.mark.remote
    @pytest.mark.planetary
    @pytest.mark.long
    @pytest.mark.skip
    def test_get_metadata_all(self):
        provider = construct_provider()
        collections = provider.available_collections()

        for collection in collections[:60]:
            metadata = provider.get_metadata(collection)
            assert metadata is None or isinstance(metadata, CollectionMetadata)
            time.sleep(1)

    @pytest.mark.remote
    @pytest.mark.planetary
    def test_get_metadata_sentinel2(self):
        provider = construct_provider()
        metadata = provider.get_metadata('sentinel-2-l2a')

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
    @pytest.mark.planetary
    def test_get_metadata_alos_palsar_mosaic(self):
        # This is a weird edge-case where planetary-computer is not STAC v1 conform!
        # For this collections, asset_def.roles does not exist, instead there is a asset_def.role property.
        # We should make a best-effort to catch this.
        provider = construct_provider()
        metadata = provider.get_metadata('alos-palsar-mosaic')
        hh = metadata.get_variable('HH')
        assert hh.roles == ['data']

    @pytest.mark.remote
    def test_get_metadata_landsat(self):
        provider = construct_provider()
        metadata = provider.get_metadata('landsat-c2-l2')

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

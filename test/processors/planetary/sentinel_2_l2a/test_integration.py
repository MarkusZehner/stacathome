import datetime

import numpy as np
import pystac
import pytest
import shapely
import stacathome
import xarray as xr
from odc.geo.geobox import GeoBox


MPI_BGC_COORDS = shapely.Point(680450.0, 5642969.0)  # EPSG:32632
NEAR_TILE_CORNER = shapely.Point(705800, 5896040)  # EPSG:32632
NEAR_TILE_CORNER_SHIFT = shapely.Point(705800 + 5000, 5896040 + 5000)  # EPSG:32632


def example_geobox(center, crs='EPSG:32632', resolution=10, size=500):
    bbox = (
        center.x - size,  # left
        center.y - size,  # bottom
        center.x + size,  # right
        center.y + size,  # top
    )  # 1km x 1km
    geobox = GeoBox.from_bbox(
        bbox=bbox,
        crs=crs,
        resolution=resolution,
        tight=True,
    )
    return geobox


class TestSentinel2L2AIntegration:

    @pytest.mark.remote
    @pytest.mark.planetary
    def test_search_items_geoboxed_nodefault(self):
        expected_ids = [
            'S2A_MSIL2A_20230106T102411_R065_T32UPB_20240807T164608',
            'S2A_MSIL2A_20230106T102411_R065_T32UPB_20230107T001757',
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20230101T222806',
        ]

        geobox = example_geobox(MPI_BGC_COORDS)
        items = stacathome.search_items_geoboxed(
            provider_name='planetary_computer',
            collection='sentinel-2-l2a',
            geobox=geobox,
            starttime='2023-01-01',
            endtime='2023-01-09',
            no_default_processor=True,
        )

        assert isinstance(items, pystac.ItemCollection)
        assert {item.id for item in items} == set(expected_ids)

    @pytest.mark.remote
    @pytest.mark.planetary
    def test_search_items_geoboxed(self):
        expected_ids = [
            'S2A_MSIL2A_20230106T102411_R065_T32UPB_20240807T164608',
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20240806T223544',
        ]

        geobox = example_geobox(MPI_BGC_COORDS)
        items = stacathome.search_items_geoboxed(
            provider_name='planetary_computer',
            collection='sentinel-2-l2a',
            geobox=geobox,
            starttime='2023-01-01',
            endtime='2023-01-09',
        )

        assert isinstance(items, pystac.ItemCollection)
        assert {item.id for item in items} == set(expected_ids)

    @pytest.mark.remote
    @pytest.mark.planetary
    def test_load_geoboxed_nodefault(self):
        # This is actually a special case:
        # We have 4 items here, but only two observations, two items are reprocessings!
        # We expect all four of them to be included in the data for the raw (no processor) load function

        expected_ids = [
            'S2A_MSIL2A_20230106T102411_R065_T32UPB_20240807T164608',
            'S2A_MSIL2A_20230106T102411_R065_T32UPB_20230107T001757',
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20240806T223544',
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20230101T222806',
        ]

        expected_vars = {
            'AOT',
            'B01',
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
            'B08',
            'B09',
            'B11',
            'B12',
            'B8A',
            'SCL',
            'WVP',
            'visual',
        }

        data_vars = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']

        geobox = example_geobox(MPI_BGC_COORDS)
        items, ds = stacathome.load_geoboxed(
            provider_name='planetary_computer',
            collection='sentinel-2-l2a',
            geobox=geobox,
            starttime='2023-01-01',
            endtime='2023-01-09',
            no_default_processor=True,
        )

        assert isinstance(items, pystac.ItemCollection)
        assert isinstance(ds, xr.Dataset)

        assert {item.id for item in items} == set(expected_ids)

        # Test if all variables are there
        assert set(ds.data_vars.keys()) == expected_vars

        # Test dimensions
        for var in expected_vars:
            assert ds[var].dims == ('time', 'y', 'x')

        # Test shape
        assert len(ds.y) == geobox.shape[0]
        assert len(ds.x) == geobox.shape[1]
        assert len(ds.time) == len(items)

        # TODO: test correct location

        # Test correct timestamps
        item_timestamps = [item.datetime.astimezone(datetime.timezone.utc) for item in items]
        item_timestamps = [np.datetime64(dt.replace(tzinfo=None)) for dt in item_timestamps]
        assert list(ds.time.values) == item_timestamps

        # Test dtypes
        for var in data_vars:
            assert ds[var].dtype == np.dtype(
                'float32'
            )  # geotiff is not stored in int16, no conversion done from our side

        # Test for plausible values in data layers
        for var in data_vars:
            data = ds[var].values
            minimum, maximum = data.min(), data.max()
            std = data.std()
            assert minimum >= -1000
            assert maximum <= 17000
            assert std >= 100.0

        # Test integer values in scene classification layer
        unique_values = np.unique(ds['SCL'].values)
        assert np.isin(unique_values, np.arange(1, 12)).all()

    @pytest.mark.remote
    @pytest.mark.planetary
    def test_load_geoboxed(self):
        # This is actually a special case:
        # We have 4 items here, but only two observations, two items are reprocessings!
        # We expect the processor to drop the old ones

        expected_ids = [
            'S2A_MSIL2A_20230106T102411_R065_T32UPB_20240807T164608',
            'S2B_MSIL2A_20230101T102339_R065_T32UPB_20240806T223544',
        ]

        expected_vars = {
            'AOT',
            'B01',
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
            'B08',
            'B09',
            'B11',
            'B12',
            'B8A',
            'SCL',
            'WVP',
            'visual',
        }

        data_vars = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']

        geobox = example_geobox(MPI_BGC_COORDS)
        items, ds = stacathome.load_geoboxed(
            provider_name='planetary_computer',
            collection='sentinel-2-l2a',
            geobox=geobox,
            starttime='2023-01-01',
            endtime='2023-01-09',
        )

        assert isinstance(items, pystac.ItemCollection)
        assert isinstance(ds, xr.Dataset)

        assert {item.id for item in items} == set(expected_ids)

        # Test if all variables are there
        assert set(ds.data_vars.keys()) == expected_vars

        # Test dimensions
        for var in expected_vars:
            assert ds[var].dims == ('time', 'y', 'x')

        # Test shape
        assert len(ds.y) == geobox.shape[0]
        assert len(ds.x) == geobox.shape[1]
        assert len(ds.time) == len(items)

        # Test correct timestamps
        item_timestamps = [item.datetime.astimezone(datetime.timezone.utc) for item in items]
        item_timestamps = [np.datetime64(dt.replace(tzinfo=None)) for dt in item_timestamps]
        assert list(ds.time.values) == item_timestamps

        # Test dtypes
        for var in data_vars:
            assert ds[var].dtype == np.dtype('uint16')

        # Test for plausible values in data layers
        for var in data_vars:
            data = ds[var].values
            minimum, maximum = data.min(), data.max()
            std = data.std()
            assert minimum >= -1000
            assert maximum <= 17000
            assert std >= 100.0

        # Test integer values in scene classification layer
        unique_values = np.unique(ds['SCL'].values)
        assert np.isin(unique_values, np.arange(1, 12)).all()

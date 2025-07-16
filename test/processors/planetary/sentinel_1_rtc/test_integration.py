import datetime
import json

import numpy as np
import odc.geo.geom as geom
import pystac
import pytest
import stacathome
import xarray as xr


TUEBINGEN = '{"type": "Polygon","coordinates": [[[8.990656040014244, 48.487788472891566],[9.141792834206825, 48.487788472891566],[9.141792834206825, 48.56039733619019], [8.990656040014244, 48.56039733619019],[8.990656040014244, 48.487788472891566]]]}'


class TestSentinel1RTCIntegration:

    @pytest.mark.remote
    @pytest.mark.planetary
    def test_load_items_nodefault(self):
        expected_ids = [
            'S1A_IW_GRDH_1SDV_20250708T053446_20250708T053511_059988_0773C3_rtc',
            'S1A_IW_GRDH_1SDV_20250704T171559_20250704T171624_059937_077200_rtc',
        ]

        expected_vars = {'vv', 'vh'}

        roi = geom.Geometry(json.loads(TUEBINGEN), crs='EPSG:4326')
        items, ds = stacathome.load(
            provider_name='planetary_computer',
            collection='sentinel-1-rtc',
            roi=roi,
            starttime='2025-07-01',
            endtime='2025-07-20',
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

        # Test correct timestamps
        item_timestamps = [item.datetime.astimezone(datetime.timezone.utc) for item in items]
        item_timestamps = [np.datetime64(dt.replace(tzinfo=None)) for dt in item_timestamps]
        assert list(ds.time.values) == item_timestamps

        # Should be fp32
        for var in ds.data_vars:
            assert ds[var].dtype == np.dtype('float32')

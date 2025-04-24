import datetime
import numpy as np
import xarray as xr
from shapely import transform, Polygon, box
from pyproj import CRS

from .base import STACItemProcessor, Band


from stacathome.utils import get_transform
from stacathome.redo_classes.generic_utils import create_utm_grid_bbox
from stacathome.sentinel_2_utils import compute_scale_and_offset


class Sentinel2L2AProcessor(STACItemProcessor):
    tilename = "s2:mgrs_tile"
    datetime_id = "datetime"
    supported_bands = [
        "B01", "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"
    ]
    special_bands = {
        'SCL' :
            Band(
                'SCL',
                'uint8',
                0,
                20,
                False,
                long_name="Scene Classification Layer",
                flag_meanings=[
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
                flag_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            )
    }

    def get_crs(self):
        return self.item.properties["proj:epsg"]

    def get_tilename_value(self):
        return self.item.properties[self.tilename]

    def get_assets_as_bands(self):
        supported_bands_S2 = self.supported_bands
        present_bands = self.get_data_asset_keys()
        present_supported_bands = sorted(set(supported_bands_S2) & set(present_bands))

        bands = []
        for b in present_supported_bands:
            special = self.special_bands.get(b, None)
            if special:
                bands.append(special)
            else:
                info = self.item.assets[b].to_dict()
                eo_bands = info.get('eo:bands')
                if isinstance(eo_bands, list) and len(eo_bands) == 1:
                    eo_bands = eo_bands[0]
                bands.append(
                    Band(
                        b,
                        "uint16",
                        0,
                        info.get('gsd', 0),
                        True,
                        long_name=eo_bands.get('description'),
                        center_wavelenght=eo_bands.get('center_wavelength'),
                        full_width_half_max=eo_bands.get('full_width_half_max'),
                    )
                )

        return bands

    def get_bbox(self):
        key = self.get_data_asset_keys()[0]
        return box(*self.item.assets[key].extra_fields['proj:bbox'])

    def get_band_attributes(self, bands: list[str] | set[str]):
        info = self.get_assets_as_bands()
        bandattributes = {}
        for i in info:
            d = i.to_dict()
            bandattributes[d['Name']] = d
        bandattributes = {b: bandattributes[b] for b in bands if b in bandattributes}
        return bandattributes

    def get_data_asset_keys(self, role="data"):
        key = list(self.item.get_assets(role=role).keys())
        assert len(key) > 0, "No data assets found!"
        return key

    def get_data_coverage_geometry(self):
        return transform(Polygon(*self.item.geometry["coordinates"]), get_transform(4326, self.get_crs()))

    def snap_bbox_to_grid(self, bbox, grid_size=60):
        def arange_bounds(bounds, step):
            return np.arange(bounds[0], bounds[2], step), np.arange(bounds[1], bounds[3], step)

        inherent_bbox = self.get_bbox()
        inherent_x, inherent_y = arange_bounds(inherent_bbox.bounds, grid_size)
        created_box = create_utm_grid_bbox(bbox.bounds, grid_size)
        created_x, created_y = arange_bounds(created_box.bounds, grid_size)

        # Determine the overlapping grid coordinates
        shared_x = set(created_x) & set(inherent_x)
        shared_y = set(created_y) & set(inherent_y)

        if len(created_x) >= len(inherent_x):
            overlap_x = created_x[(created_x >= min(inherent_x)) & (created_x <= max(inherent_x))]
            overlap_y = created_y[(created_y >= min(inherent_y)) & (created_y <= max(inherent_y))]

        else:
            overlap_x = inherent_x[(inherent_x >= min(created_x)) & (inherent_x <= max(created_x))]
            overlap_y = inherent_y[(inherent_y >= min(created_y)) & (inherent_y <= max(created_y))]

        if len(shared_x) != len(overlap_x) or len(shared_y) != len(overlap_y):
            raise ValueError("Grid of created box does not fully align with inherent grid")

        return created_box

    def load_cube(self, items, bands, geobox, provider, path, savefunc, split_by=100, chunks: dict = None):
        # most of this until the s2 specific harmonization could be moved to the base class

        # implement this in multiprocess?

        if len(items) > 100:
            raise Warning('large amount of assets, consider loading split in smaller time steps!')

        parameters = {
            "groupby": "solar_day",
            "fail_on_error": True,
        }

        if chunks:
            assert set(chunks.keys()) == {"time", "x", "y"}, "Chunks must contain the dimensions 'time', 'x', 'y'!"
            parameters['chunks'] = chunks

        multires_cube = {}
        for gb, band_subset in geobox.items():
            req_bands = set(band_subset) & set(bands)
            resampling = self.get_resampling_per_band(gb.resolution.x)
            dtypes = self.get_dtype_per_band(gb.resolution.x)
            resampling = {k: resampling[k] for k in req_bands if k in resampling}
            dtypes = {k: dtypes[k] for k in req_bands if k in dtypes}

            parameters['bands'] = req_bands
            parameters['geobox'] = gb
            parameters['resampling'] = resampling
            parameters['dtype'] = dtypes
            if split_by is not None and len(items) > split_by:
                split_items = self.split_items_keep_solar_days_together(items, split_by)
                cube = []
                for split in split_items:
                    parameters['items'] = split
                    cube.append(provider.download_cube(parameters))
                cube = xr.concat(cube, dim="time")

            else:
                parameters['items'] = items
                cube = provider.download_cube(parameters)
            attrs = self.get_band_attributes(req_bands)
            for band in cube.keys():
                cube[band].attrs = attrs[band]

            multires_cube[int(gb.resolution.x)] = cube

        # remove empty images, could be moved into separate function
        # get coarsest resolution first band and check where mean value is 0:
        coarsest = max(multires_cube.keys())
        first_var = list(multires_cube[coarsest].data_vars.keys())[0]
        mean_over_time = multires_cube[coarsest][first_var].mean(dim=['x', 'y'])
        na_value = multires_cube[coarsest][first_var].attrs['NoData Value']
        mask_over_time = np.where(mean_over_time != na_value)[0]

        chunking = {"time": 2 if not chunks else chunks['time']}
        for spat_res in multires_cube.keys():
            # remove empty images, could be moved into separate function
            multires_cube[spat_res] = multires_cube[spat_res].isel(time=mask_over_time)

            chunking[f'x{spat_res}'] = -1 if not chunks else chunks['x']
            chunking[f'y{spat_res}'] = -1 if not chunks else chunks['y']
            multires_cube[spat_res] = multires_cube[spat_res].rename({'x': f'x{spat_res}', 'y': f'y{spat_res}'})

        multires_cube = xr.merge(multires_cube.values())

        # mask by SCL?
        # remove values time steps with no data per SCL layer?

        # this changes int16 to float32 could also be seen as different function
        multires_cube = self.harmonize_s2_data(multires_cube, scale=True)

        multires_cube = multires_cube.chunk(chunking)

        for i in multires_cube.data_vars.keys():
            if i not in ['SCL', 'spatial_ref']:
                multires_cube[i] = multires_cube[i].astype("float32")
                multires_cube[i].encoding = {"dtype": "uint16",
                                             "scale_factor": compute_scale_and_offset(multires_cube[i].values),
                                             "add_offset": 0.0,
                                             "_FillValue": 65535}
            elif i == 'SCL':
                multires_cube[i] = multires_cube[i].astype("uint8")

        return multires_cube

    def harmonize_s2_data(self, data: xr.Dataset, scale: bool = True) -> xr.Dataset:
        """
        Harmonize new Sentinel-2 data to the old baseline. Data after 25-01-2022 is clipped
        to 1000 and then subtracted by 1000.
        From https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change
        adjusted to odc-stac, using different variables for each band.

        Parameters
        ----------
        data: xarray.DataArray
            A DataArray with four dimensions: time, band, y, x

        Returns
        -------
        harmonized: xarray.DataArray
            A DataArray with all values harmonized to the old
            processing baseline.
        """
        cutoff = datetime.datetime(2022, 1, 25)
        offset = 1000

        bands = self.get_data_asset_keys()
        if "SCL" in bands:
            bands.remove("SCL")

        to_process = list(set(bands) & set(data.keys()))

        attrs = {p: data[p].attrs for p in to_process}

        no_change = data.drop_vars(to_process)
        old = data[to_process].sel(time=slice(cutoff))
        new_harmonized = data[to_process].sel(time=slice(cutoff, None)).clip(offset)
        new_harmonized -= offset

        new = xr.concat([old, new_harmonized], "time")
        if scale:
            new = new.where(new != 0)
            new = (new * 0.0001).astype("float32")
        else:
            new = new.astype("uint16")

        for variable in no_change.keys():
            new[variable] = no_change[variable]

        for a in attrs.keys():
            if scale:
                attrs[a]['Data Type'] = "float32"
                attrs[a]['NoData Value'] = np.nan
                attrs[a]['Harmonized'] = 'True'
            new[a].attrs = attrs[a]

        return new


class Modis13Q1Processor(STACItemProcessor):
    tilename = "modis:tile-id"
    datetime_id = "start_datetime"

    indices = ["250m_16_days_NDVI",
               "250m_16_days_EVI"]

    reflectances = ["250m_16_days_MIR_reflectance",
                    "250m_16_days_NIR_reflectance",
                    "250m_16_days_red_reflectance",
                    "250m_16_days_blue_reflectance"]

    indices_bands = {
        i :
            Band(
                name=i,
                data_type='int16',
                nodata_value=-3000,
                spatial_resolution=250,
                continuous_measurement=True,
                # kwargs
                scale_factor=0.0001,
                valid_range=(-2000, 10000),
            ) for i in indices
    }

    reflectance_bands = {
        i :
            Band(
                name=i,
                data_type='int16',
                nodata_value=-1000,
                spatial_resolution=250,
                continuous_measurement=True,
                # kwargs
                scale_factor=0.0001,
                valid_range=(0, 10000),
            ) for i in reflectances
    }

    other_bands = {
        "250m_16_days_VI_Quality" :
            Band(
                name="250m_16_days_VI_Quality",
                data_type='uint16',
                nodata_value=65535,
                spatial_resolution=250,
                continuous_measurement=True,
                # kwargs
                valid_range=(0, 65534)
            ),
        "250m_16_days_pixel_reliability" :
            Band(
                name="250m_16_days_pixel_reliability",
                data_type='int16',
                nodata_value=-1,
                spatial_resolution=250,
                continuous_measurement=False,
                # kwargs
                valid_range=(0, 3)
            ),
        "250m_16_days_relative_azimuth_angle" :
            Band(
                name="250m_16_days_relative_azimuth_angle",
                data_type='int16',
                nodata_value=-4000,
                spatial_resolution=250,
                continuous_measurement=True,
                # kwargs
                scale_factor=0.01,
                valid_range=(-18000, 18000)
            ),
    }

    special_bands = {**reflectance_bands, **indices_bands, **other_bands}
    supported_bands = list(special_bands.keys())
    raise NotImplementedError("Sentinel3SynergyProcessor is not implemented yet.")

    def get_crs(self):
        return CRS.from_wkt(self.item.properties['proj:wkt2'])

    def get_tile_id(self):
        return self.item.properties[self.tilename]

    def get_bbox(self):
        bbox = Polygon(self.item.properties['proj:geometry']['coordinates'][0])
        tr = get_transform(CRS.from_wkt(self.item.asset.properties['proj:wkt2']), 4326)
        return transform(bbox, tr)


class ESAWorldCoverProcessor(STACItemProcessor):
    tilename = "esa_worldcover:product_tile"

    special_bands = {
        "map":
            Band(
                name="map",
                data_type="uint8",
                nodata_value=0,
                spatial_resolution=10,
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
                spatial_resolution=60,
                continuous_measurement=False,
                long_name="Input Quality Map"
            )
    }

    supported_bands = list(special_bands.keys())
    raise NotImplementedError("Sentinel3SynergyProcessor is not implemented yet.")

    def get_crs(self):
        return self.item.properties['proj:epsg']

    def get_tilename_value(self):
        return self.item.properties[self.tilename]

    def get_bbox(self):
        bbox = box(*self.item.bbox)
        tr = get_transform(self.item.properties['proj:epsg'], 4326)
        return transform(bbox, tr)


class Sentinel3SynergyProcessor(STACItemProcessor):
    tilename = None
    raise NotImplementedError("Sentinel3SynergyProcessor is not implemented yet.")

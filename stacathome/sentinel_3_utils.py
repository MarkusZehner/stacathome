import os
import fsspec
import xarray as xr
from shapely import Point
import zarr

from .request import probe_request
from .utils import parse_dec_to_lon_lat_point, parse_dms_to_lon_lat_point
from .download import get_asset


def get_s3_cube(probe_dict, fname, workdir):
    keys = [
        'syn-amin', 'syn-flags', 'syn-ato550',
        #'tiepoints-olci',
        #'tiepoints-meteo',
        #'tiepoints-slstr-n',
        #'tiepoints-slstr-o',
        'syn-angstrom-exp550', 
        'syn-s1n-reflectance', 'syn-s1o-reflectance', 'syn-s2n-reflectance', 
        'syn-s2o-reflectance', 'syn-s3n-reflectance', 'syn-s3o-reflectance', 'syn-s5n-reflectance', 
        'syn-s5o-reflectance', 'syn-s6n-reflectance', 'syn-s6o-reflectance', 'syn-oa01-reflectance', 
        'syn-oa02-reflectance', 'syn-oa03-reflectance', 'syn-oa04-reflectance', 'syn-oa05-reflectance', 
        'syn-oa06-reflectance', 'syn-oa07-reflectance', 'syn-oa08-reflectance', 'syn-oa09-reflectance', 
        'syn-oa10-reflectance', 'syn-oa11-reflectance', 'syn-oa12-reflectance', 'syn-oa16-reflectance', 
        'syn-oa17-reflectance', 'syn-oa18-reflectance', 'syn-oa21-reflectance', 
        #'syn-sdr-removed-pixels', 
        #'annotations-removed-pixels'
    ]

    for i in range(len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0])):
        print(f"Processing {i+1}/{len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0])}", flush=True)
        
        folder = probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0][i].assets['geolocation'].href.split('/')[-2]
        out_path_zarr_zip = os.path.join(workdir, fname, f'{folder}.zarr.zip')
        if os.path.exists(out_path_zarr_zip):
            continue
        
        for a in probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0][i].assets:
            href = probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0][i].assets[a].href
            out_path = os.path.join(workdir, fname, href.split('/')[-2], '_'.join(href.split('/')[-1:]))
            get_asset(href, out_path)
            probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0][i].assets[a].href = out_path
        
        item = probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0][i]
        geo = xr.open_dataset(fsspec.open(item.assets["geolocation"].href).open())
        folder = item.assets['geolocation'].href.split('/')[-2]
        def read(key: str) -> xr.Dataset:
            dataset = xr.open_dataset(fsspec.open(item.assets[key].href).open())
            dataset = dataset.assign_coords(
                {
                    "lat": geo.lat,
                    "lon": geo.lon,
                }
            )
            return dataset

        datasets = [read(key) for key in keys]
        dataset = xr.combine_by_coords(datasets, join="exact", combine_attrs="drop_conflicts")
        dataset = dataset.where((dataset.lat > probe_dict['sentinel-3-synergy-syn-l2-netcdf'][1].bounds[1]) & 
              (dataset.lat < probe_dict['sentinel-3-synergy-syn-l2-netcdf'][1].bounds[3]) &
              (dataset.lon > probe_dict['sentinel-3-synergy-syn-l2-netcdf'][1].bounds[0]) &
              (dataset.lon < probe_dict['sentinel-3-synergy-syn-l2-netcdf'][1].bounds[2]), drop=True)
        
        store = zarr.ZipStore(out_path_zarr_zip, mode="x")
        dataset.to_zarr(store, mode="w-")
        store.close()

if __name__ == '__main__':
    collection = "sentinel-3-synergy-syn-l2-netcdf"
    edge_length_m = 4600
    target_res_m = 300
    flux_test_pos = [
        # ('AU-Dry', '-15.2588, 132.3706'),
        # ('AU-How', '-12.4943, 131.1523'),
        # ('BE-Lon', '50.5516, 4.7462'),
        # ('CD-Ygb', '0.8144, 24.5025'),
        # ('CH-Dav', '46.8153, 9.8559'),
        # ('CZ-Lnz', '48.6816, 16.9464'),
        # ('DE-Hai', '51.0792, 10.4530'),
        # ('DE-RuR', '50.6219, 6.3041'),
        # ('DE-Tha', '50.9626, 13.5651'),
        # ('ES-LMa', '39.9415, -5.7734'),
        # ('FR-Fon', '48.4764, 2.7801'),
        # ('GF-Guy', '5.2788, -52.9249'),
        # ('IT-Noe', '40.6062, 8.1512'),
        # ('US-Rpf', '65.1198, -147.4290'),
        # ('US-SRG', '31.7894, -110.8277'),
        # ('US-Tw4', '38.1027, -121.6413'),
        # ('US-UMB', '45.5598, -84.7138'),
        # ('US-UMd', '45.5625, -84.6975'),
        # ('US-Var', '38.4133, -120.9507'),
        # ('US-xDS', '28.1250, -81.4362'),
        ]
    
    workdir = '/Net/Groups/BGI/work_4/scratch/jnelson/Sen3_cutouts'
    time_range = {
        collection: (2018, 2025),
                    }
    
    for fname, position_str in flux_test_pos:
        try:
            position = parse_dms_to_lon_lat_point(position_str)
        except TypeError:
            pass
        try:
            position = parse_dec_to_lon_lat_point(position_str)
        except (TypeError, ValueError):
            pass
        assert position is not None
        assert isinstance(position, Point)
    
        print(f"Processing {fname} at {position}", flush=True)
        probe_dict = probe_request(
            point_wgs84=position, 
            distance_in_m=edge_length_m, 
            collection=[collection], 
            return_box=True, limit=9999
        )        
        print(f"Found {len(probe_dict[collection][0])} scenes", flush=True)
        get_s3_cube(probe_dict, fname, workdir)

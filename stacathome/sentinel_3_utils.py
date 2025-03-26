import os

import fsspec
import xarray as xr
import zarr
# from shapely import Point
import numpy as np
# import pyinterp
# from shapely import box, transform, buffer
import zipfile
from copy import copy
import json
from pystac import Item

# from concurrent.futures import ThreadPoolExecutor

from .download import download_assets_parallel
# from .request import probe_request
# from .utils import parse_dec_to_lon_lat_point, parse_dms_to_lon_lat_point,
# from .utils import get_utm_crs_from_lon_lat, get_transform  # , compute_scale_and_offset


def process_timestep(t_index, item, keys, Y0, Y1, X0, X1, mesh, meshstack, resampled_Y, resampled_X):
    print(t_index, item.properties['datetime'], flush=True)

    mesh = copy(mesh)

    with xr.open_dataset(fsspec.open(item.assets["geolocation"].href).open()) as geo:
        lats = geo['lat'].values.flatten()
        lons = geo['lon'].values.flatten()

    mask = np.where((lats > Y0) & (lats < Y1) & (lons > X0) & (lons < X1))

    if len(mask[0]) == 0:
        return t_index, np.zeros((len(keys) * 2 + 1, len(resampled_Y), len(resampled_X)), dtype=np.float32), []

    lats, lons = lats[mask], lons[mask]
    lonlatstack = np.vstack((lons, lats)).T

    index_map = np.arange(len(lonlatstack))
    mesh.packing(lonlatstack, index_map)
    idw_index, _ = mesh.inverse_distance_weighting(
        meshstack, within=False, k=1, num_threads=1
    )
    idw_index = idw_index.astype(np.int32)

    local_varlist = []
    local_values = np.zeros((len(keys) * 2 + 1, len(resampled_Y), len(resampled_X)), dtype=np.float32)

    pointer_var = 0
    for k in keys:
        with xr.open_dataset(fsspec.open(item.assets[k].href).open()) as dataset:
            for var in dataset.data_vars:
                if t_index == 0:
                    local_varlist.append(var)

                data = dataset[var].values.flatten()[mask]
                idw = data[idw_index].reshape((len(resampled_Y), len(resampled_X)))
                local_values[pointer_var] = idw
                pointer_var += 1

    return t_index, local_values, local_varlist


def load_s3_cube(probe_dict, fname, workdir):
    # keys = [
    #     'syn-amin',
    #     'syn-flags',
    #     'syn-ato550',
    #     # 'tiepoints-olci',
    #     # 'tiepoints-meteo',
    #     # 'tiepoints-slstr-n',
    #     # 'tiepoints-slstr-o',
    #     'syn-angstrom-exp550',
    #     'syn-s1n-reflectance',
    #     'syn-s1o-reflectance',
    #     'syn-s2n-reflectance',
    #     'syn-s2o-reflectance',
    #     'syn-s3n-reflectance',
    #     'syn-s3o-reflectance',
    #     'syn-s5n-reflectance',
    #     'syn-s5o-reflectance',
    #     'syn-s6n-reflectance',
    #     'syn-s6o-reflectance',
    #     'syn-oa01-reflectance',
    #     'syn-oa02-reflectance',
    #     'syn-oa03-reflectance',
    #     'syn-oa04-reflectance',
    #     'syn-oa05-reflectance',
    #     'syn-oa06-reflectance',
    #     'syn-oa07-reflectance',
    #     'syn-oa08-reflectance',
    #     'syn-oa09-reflectance',
    #     'syn-oa10-reflectance',
    #     'syn-oa11-reflectance',
    #     'syn-oa12-reflectance',
    #     'syn-oa16-reflectance',
    #     'syn-oa17-reflectance',
    #     'syn-oa18-reflectance',
    #     'syn-oa21-reflectance',
    #     # 'syn-sdr-removed-pixels',
    #     # 'annotations-removed-pixels'
    # ]
    to_process = []
    for i in range(len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0])):

        folder = probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0][i].assets['geolocation'].href.split('/')[-2]
        out_path_zarr_zip = os.path.join(workdir, fname, f'{folder}.zarr.zip')
        if os.path.exists(out_path_zarr_zip):
            continue
        print(f"Loading {fname} {i + 1}/{len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0])}", flush=True)
        for a in probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0][i].assets:
            href = probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0][i].assets[a].href
            out_path = os.path.join(workdir, fname, href.split('/')[-2], '_'.join(href.split('/')[-1:]))
            if not os.path.exists(out_path):
                to_process.append((href, out_path))
                # get_asset(href, out_path)
            probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0][i].assets[a].href = out_path
    workers = int(os.getenv("SLURM_CPUS_PER_TASK", 4))
    download_assets_parallel(to_process, max_workers=workers)

    out_path_query = os.path.join(workdir, fname, f"{fname}_S3_query.json")
    with open(out_path_query, "w") as json_file:
        json.dump(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0].to_dict(), json_file, indent=4)

    return probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]

    # regridding step

    # crs_code = get_utm_crs_from_lon_lat(position.x, position.y)
    # tr = get_transform(4326, crs_code)
    # tr_back = get_transform(crs_code, 4326)
    # point_utm = transform(position, tr)
    # distance_in_m = edge_length
    # bbox_wgs84 = transform(box(*buffer(point_utm, distance_in_m / 2).bounds), tr_back)

    # X0, Y0, X1, Y1 = bbox_wgs84.bounds

    # mesh = pyinterp.RTree()
    # resampled_X = np.linspace(X0, X1, int(distance_in_m / target_res))
    # resampled_Y = np.linspace(Y0, Y1, int(distance_in_m / target_res))

    # mx, my = np.meshgrid(resampled_X,
    #                      resampled_Y,
    #                      indexing='ij')
    # meshstack = np.vstack((mx.ravel(), my.ravel())).T

    # timeax = np.array(
    #     [np.datetime64(item.properties['datetime'][:-1])
    #      for item in probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]])

    # orbit_dir = np.array([item.properties['sat:orbit_state'] for item in probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]])
    # abs_orbit = np.array([item.properties['sat:absolute_orbit'] for item in probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]])
    # rel_orbit = np.array([item.properties['sat:relative_orbit'] for item in probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]])

    # values = np.zeros(
    #     (len(keys) * 2 + 1,
    #      len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]),
    #      len(resampled_Y),
    #      len(resampled_X)),
    #     dtype=np.float32)

    # # Use ThreadPoolExecutor to parallelize over timesteps
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     results = list(executor.map(
    #         process_timestep,
    #         range(len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0])),
    #         probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0],
    #         [keys] * len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]),
    #         [Y0] * len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]),
    #         [Y1] * len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]),
    #         [X0] * len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]),
    #         [X1] * len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]),
    #         [mesh] * len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]),
    #         [meshstack] * len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]),
    #         [resampled_Y] * len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]),
    #         [resampled_X] * len(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0])))

    # # t_index, item, keys, Y0, Y1, X0, X1, mesh, meshstack, resampled_Y, resampled_X

    # # Aggregate results
    # for t_index, local_values, local_varlist in results:
    #     values[:, t_index] = local_values
    #     if t_index == 0:
    #         varlist = local_varlist

    # regridding without parallelization
    # varlist = []
    # for t_index, item in enumerate(probe_dict['sentinel-3-synergy-syn-l2-netcdf'][0]):
    #     mesh = copy(mesh)
    #     print(t_index, item.properties['datetime'], flush=True)

    #     geo = xr.open_dataset(fsspec.open(item.assets["geolocation"].href).open())
    #     lats = geo['lat'].values.flatten()
    #     lons = geo['lon'].values.flatten()

    #     mask = np.where((lats > Y0) & (lats < Y1) & (lons > X0) & (lons < X1))

    #     lats, lons = lats[mask], lons[mask]
    #     lonlatstack = np.vstack((lons, lats)).T  # Precomputed for all k

    #     # Use np.arange to create an index mapping
    #     index_map = np.arange(len(lonlatstack))
    #     mesh.packing(lonlatstack, index_map)
    #     idw_index, _ = mesh.inverse_distance_weighting(
    #         meshstack,
    #         within=False,  # No extrapolation
    #         k=1,  # Find 1 nearest neighbor
    #         num_threads=1,
    #     )
    #     idw_index = idw_index.astype(np.int32)
    #     pointer_var = 0
    #     for k in keys:
    #         dataset = xr.open_dataset(fsspec.open(item.assets[k].href).open())
    #         for var in list(dataset.data_vars):
    #             if t_index == 0:
    #                 varlist.append(var)

    #             data = dataset[var].values.flatten()[mask]
    #             idw = data[idw_index].reshape(mx.shape)
    #             values[pointer_var, t_index] = idw

    #             pointer_var += 1

    # ds = xr.Dataset(
    #     coords={
    #         "lon": resampled_X,
    #         "lat": resampled_Y,
    #         "time": timeax.astype('datetime64[ns]')
    #     },
    #     attrs={"crs": 4326,
    #            "Dataset": "Sentinel-3 OLCI Level 2 data regridded"}
    # )
    # ds['orbit_state'] = xr.DataArray(
    #     orbit_dir,
    #     dims=("time"),
    # )
    # ds['abs_orbit'] = xr.DataArray(
    #     abs_orbit,
    #     dims=("time"),
    # )
    # ds['rel_orbit'] = xr.DataArray(
    #     rel_orbit,
    #     dims=("time"),
    # )

    # for i, k in enumerate(varlist):
    #     ds[k] = xr.DataArray(
    #         values[i],
    #         dims=("time", "lat", "lon"),
    #     )

    # ds = ds.sortby('time')

    # for i in list(ds.data_vars):
    #     if i not in ['orbit_state', 'abs_orbit', 'rel_orbit']:
    #         ds[i] = ds[i].astype("float32")
    #         ds[i].encoding = {"dtype": "uint16",
    #                           "scale_factor": compute_scale_and_offset(ds[i].values),
    #                           "add_offset": 0.0,
    #                           "_FillValue": 65535}

    # out_path = os.path.join(workdir, fname, f"{fname}_S3.zarr.zip")
    # store = zarr.ZipStore(out_path, mode="w", compression=zipfile.ZIP_BZIP2)
    # ds.to_zarr(store, mode="w", consolidated=True)
    # store.close()


def cut_s3_cubes(query_dict, fname, position, edge_length, workdir):
    keys = [
        'syn-amin',
        'syn-flags',
        'syn-ato550',
        # 'tiepoints-olci',
        # 'tiepoints-meteo',
        # 'tiepoints-slstr-n',
        # 'tiepoints-slstr-o',
        'syn-angstrom-exp550',
        'syn-s1n-reflectance',
        'syn-s1o-reflectance',
        'syn-s2n-reflectance',
        'syn-s2o-reflectance',
        'syn-s3n-reflectance',
        'syn-s3o-reflectance',
        'syn-s5n-reflectance',
        'syn-s5o-reflectance',
        'syn-s6n-reflectance',
        'syn-s6o-reflectance',
        'syn-oa01-reflectance',
        'syn-oa02-reflectance',
        'syn-oa03-reflectance',
        'syn-oa04-reflectance',
        'syn-oa05-reflectance',
        'syn-oa06-reflectance',
        'syn-oa07-reflectance',
        'syn-oa08-reflectance',
        'syn-oa09-reflectance',
        'syn-oa10-reflectance',
        'syn-oa11-reflectance',
        'syn-oa12-reflectance',
        'syn-oa16-reflectance',
        'syn-oa17-reflectance',
        'syn-oa18-reflectance',
        'syn-oa21-reflectance',
        # 'syn-sdr-removed-pixels',
        # 'annotations-removed-pixels'
    ]

    if isinstance(query_dict, Item):
        query_dict = [query_dict]

    for i, item in enumerate(query_dict):
        out_path_zarr_zip = os.path.join(workdir, fname, f"{item.assets['geolocation'].href.split('/')[-2]}.zarr.zip")
        if os.path.exists(out_path_zarr_zip):
            # print('exists')
            continue

        print(f"{fname} {i + 1}/{len(query_dict)}: {out_path_zarr_zip}", flush=True)
        geo = xr.open_dataset(item.assets['geolocation'].href).compute()
        _dist = ((position.y - geo.lat.values) ** 2 + (position.x - geo.lon.values) ** 2)
        idx, idy = np.where(_dist == _dist.min())
        if len(idx) == 0 or len(idy) == 0:
            continue
        idy = idy.tolist()[0]
        idx = idx.tolist()[0]
        dist = edge_length // 2

        min_x = idx - dist if idx - dist >= 0 else 0
        max_x = idx + dist if idx + dist < geo.sizes['rows'] else geo.sizes['rows']
        min_y = idy - dist if idy - dist >= 0 else 0
        max_y = idy + dist if idy + dist < geo.sizes['columns'] else geo.sizes['columns']
        x_range = slice(min_x, max_x)
        y_range = slice(min_y, max_y)

        geo = geo.isel(rows=x_range, columns=y_range)

        geo = geo.assign_coords(
            {
                "lat": geo.lat,
                "lon": geo.lon,
            }
        )
        for k in keys:
            dataset = xr.open_dataset(fsspec.open(item.assets[k].href).open())
            dataset = dataset.isel(rows=x_range, columns=y_range)
            dataset = dataset.assign_coords(
                {
                    "lat": geo.lat,
                    "lon": geo.lon,
                }
            )
            for var in list(dataset.data_vars):
                geo[var] = dataset[var]
        store = zarr.ZipStore(out_path_zarr_zip, mode="w", compression=zipfile.ZIP_BZIP2)
        geo.to_zarr(store, mode="w", consolidated=True)
        store.close()

    # if __name__ == '__main__':
    #     collection = "sentinel-3-synergy-syn-l2-netcdf"
    #     edge_length_m = 3000
    #     target_res_m = 300
    #     flux_test_pos = [
    #         ('AU-Dry', '-15.2588, 132.3706'),
    #         # ('AU-How', '-12.4943, 131.1523'),
    #         # ('BE-Lon', '50.5516, 4.7462'),
    #         # ('CD-Ygb', '0.8144, 24.5025'),
    #         # ('CH-Dav', '46.8153, 9.8559'),
    #         # ('CZ-Lnz', '48.6816, 16.9464'),
    #         # ('DE-Hai', '51.0792, 10.4530'),
    #         # ('DE-RuR', '50.6219, 6.3041'),
    #         # ('DE-Tha', '50.9626, 13.5651'),
    #         # ('ES-LMa', '39.9415, -5.7734'),
    #         # ('FR-Fon', '48.4764, 2.7801'),
    #         # ('GF-Guy', '5.2788, -52.9249'),
    #         # ('IT-Noe', '40.6062, 8.1512'),
    #         # ('US-Rpf', '65.1198, -147.4290'),
    #         # ('US-SRG', '31.7894, -110.8277'),
    #         # ('US-Tw4', '38.1027, -121.6413'),
    #         # ('US-UMB', '45.5598, -84.7138'),
    #         # ('US-UMd', '45.5625, -84.6975'),
    #         # ('US-Var', '38.4133, -120.9507'),
    #         # ('US-xDS', '28.1250, -81.4362'),
    #     ]

    #     workdir = '/Net/Groups/BGI/work_4/scratch/jnelson/Sen3_cutouts'
    #     time_range = {
    #         collection: (2018, 2025),
    #     }

    #     for fname, position_str in flux_test_pos:
    #         try:
    #             position = parse_dms_to_lon_lat_point(position_str)
    #         except TypeError:
    #             pass
    #         try:
    #             position = parse_dec_to_lon_lat_point(position_str)
    #         except (TypeError, ValueError):
    #             pass
    #         assert position is not None
    #         assert isinstance(position, Point)

    #         print(f"Processing {fname} at {position}", flush=True)
    #         probe_dict = probe_request(
    #             point_wgs84=position, distance_in_m=edge_length_m, collection=[collection], return_box=True, limit=9999
    #         )
    #         print(f"Found {len(probe_dict[collection][0])} scenes", flush=True)
    #         get_s3_cube(probe_dict, fname, position, edge_length_m, target_res_m, workdir)

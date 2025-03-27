import datetime
import os
from collections import Counter

import xarray as xr
import zarr
import numpy as np

from .asset_specs import add_attributes
from .sentinel_2_utils import drop_no_data_s2, harmonize_to_old, compute_scale_and_offset


def preprocess_collection(xds, collection):
    if collection == 'sentinel-2-l2a':
        return harmonize_to_old(drop_no_data_s2(xds))
    elif collection == 'esa-worldcover':
        if len(xds.time.values) == 1:
            xds = xds.rename_vars({"map": f"esa_worldcover_{xds.time.values.astype('datetime64[Y]')[0]}"})
        return xds
    else:
        return xds


def open_mf_zarr_zip(collection, cube_parts_paths):
    xds = xr.concat([xr.open_zarr(f) for f in cube_parts_paths], dim='time')
    xds = xds.sortby("time")
    xds = preprocess_collection(xds, collection)

    if 'time' in xds.dims and len(xds.time) > 1:
        xds = xds.rename({'time': f'time_{collection}'})
    else:
        xds = xds.squeeze().drop_vars(["time"])
    return xds


def combine_to_cube(files_to_combine, center_point, time_range, probe_dict, request_from_probe, workdir, edge_length_m, fname=None):  # , remove_parts=True):

    boxes = [request_from_probe[k][2] for k in request_from_probe.keys()]
    box_counter = Counter(boxes)
    most_common_box = box_counter.most_common(1)[0][0].boundingbox

    loc_names = [(k, request_from_probe[k][0]) for k in request_from_probe.keys()]
    comb_cubes = {}
    fname_final = []
    for k, i in loc_names:
        fname_final.append(i)
        t_files = [os.path.join(workdir, f) for f in files_to_combine if i + k in f and f.endswith('.zarr.zip')]
        comb_cubes[k] = open_mf_zarr_zip(k, t_files)

        for k in comb_cubes.keys():
            comb_cubes[k].rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True).rio.write_crs(
                most_common_box.crs, inplace=True
            )
            comb_cubes[k] = comb_cubes[k].rio.clip_box(*most_common_box, crs=most_common_box.crs)

    comb_cube = xr.merge(comb_cubes.values())

    cube_attrs = {
        'title': 'Mini Cube',
        'CRS': str(most_common_box.crs),
        'datasets_tiles': {},
        'datasets_times': {},
        'edge_length [m]': edge_length_m,
        'Creation Time': datetime.datetime.now().isoformat(),
    }

    chunking = {"x": 125, "y": 125}
    for k, _ in loc_names:
        if f'time_{k}' in comb_cube.dims:
            chunking[f'time_{k}'] = -1
        cube_attrs['datasets_tiles'][k] = probe_dict[k][0]
        cube_attrs['datasets_times'][k] = time_range[k]

    comb_cube = add_attributes(comb_cube)
    comb_cube.attrs = cube_attrs
    comb_cube = comb_cube.chunk(chunking)

    for i in comb_cube.data_vars.keys():
        if 'chunks' in comb_cube[i].encoding:
            del comb_cube[i].encoding['chunks']
        if 'preferred_chunks' in comb_cube[i].encoding:
            del comb_cube[i].encoding['preferred_chunks']

        if i not in ['SCL', 'spatial_ref']:
            comb_cube[i] = comb_cube[i].astype("float32")
            comb_cube[i].encoding = {"dtype": "uint16",
                                     "scale_factor": compute_scale_and_offset(comb_cube[i].values),
                                     "add_offset": 0.0,
                                     "_FillValue": 65535}
        elif i == 'SCL':
            comb_cube[i] = comb_cube[i].astype("uint8")
    name = fname if fname else str(np.unique(fname_final)[0])
    out_path = os.path.join(workdir, f"{name}combined.zarr.zip")

    store = zarr.ZipStore(out_path, mode="x")
    comb_cube.to_zarr(store, mode="w-", consolidated=True)
    store.close()

    # if remove_parts:
    #     for i in t_files:
    #         os.remove(i)

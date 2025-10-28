import h5py as h5
import numpy as np
import os, re
import rasterio
import xarray as xr
import rioxarray
import datetime as dt
import re


def get_datetime_from_filename(fname):
    m = re.search(r"A(\d{4})(\d{3})", fname)
    if not m:
        raise ValueError(f"Could not parse date from {fname}")
    year, doy = map(int, m.groups())
    return dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1)


def parse_struct_metadata(text):
    def grab(pat):
        m = re.search(pat, text)
        return m.group().split("=")[1] if m else None

    projection = grab(r"Projection=\w*")
    west, north = map(float, grab(r"UpperLeftPointMtrs=.*").strip("()").split(","))
    east, south = map(float, grab(r"LowerRightMtrs=.*").strip("()").split(","))
    x_dim = int(grab(r"XDim=\d*"))
    y_dim = int(grab(r"YDim=\d*"))
    return projection, west, east, south, north, x_dim, y_dim


def _coords_from_transform(transform, width, height):
    """Return x and y coordinate arrays from affine transform."""
    x = np.arange(width) * transform.a + (transform.c + transform.a / 2.0)
    y = np.arange(height) * transform.e + (transform.f + transform.e / 2.0)
    return x, y


def load_viirs_as_xarray(path, scale_and_clip:bool = False):
    with h5.File(path, "r") as file:
        mtl = file["HDFEOS INFORMATION/StructMetadata.0"][()]
        if isinstance(mtl, bytes):
            mtl = mtl.decode("utf-8")

        projection, west, east, south, north, x_dim, y_dim = parse_struct_metadata(mtl)

        if projection == "HE5_GCTP_GEO":
            crs = rasterio.CRS.from_epsg(4326)
            if any([-180 < v > 180 for v in [west, north, east, south]]):
                west, north, east, south = west/1e6, north/1e6, east/1e6, south/1e6
        else:
            crs = rasterio.CRS.from_proj4(
                "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
                "+R=6371007.181 +units=m +no_defs=True"
            )

        # Precompute transforms
        transform_500 = rasterio.transform.from_bounds(west, south, east, north, y_dim*2, x_dim*2)
        transform_1k = rasterio.transform.from_bounds(west, south, east, north, y_dim, x_dim)

        # Precompute coords
        x500, y500 = _coords_from_transform(transform_500, x_dim*2, y_dim*2)
        x1k, y1k = _coords_from_transform(transform_1k, x_dim, y_dim)

        data_vars = {}

        def collect(name, node):
            if not isinstance(node, h5.Dataset) or node.ndim != 2:
                return
            varname = os.path.basename(name)

            # read attrs
            fillvalue = node.attrs.get("_FillValue")
            offset = node.attrs.get("add_offset", node.attrs.get("offset"))
            scale = node.attrs.get("scale_factor")
            vrange = node.attrs.get("valid_range")

            if isinstance(vrange, np.ndarray):
                vmin, vmax = vrange
            elif isinstance(vrange, np.bytes_):
                vmin, vmax = map(float, vrange.decode().replace(" ", "").split("-"))
            else:
                vmin = node.attrs.get("valid_min", [None])[0]
                vmax = node.attrs.get("valid_max", [None])[0]

            # if None in [fillvalue, offset, scale, vmin, vmax]:
            #     print(f'skipping {varname}')
            #     return

            # unwrap scalars
            if isinstance(fillvalue, np.ndarray): fillvalue = fillvalue[0]
            if isinstance(offset, np.ndarray): offset = offset[0]
            if isinstance(scale, np.ndarray): scale = scale[0]

            arr = node[:]
            if scale_and_clip and not None in [fillvalue, offset, scale, vmin, vmax]:
                arr = arr.astype(np.float32) * scale + offset
                arr[(arr < vmin) | (arr > vmax) | (arr == fillvalue)] = np.nan

            h, w = arr.shape
            if (h, w) == (y_dim, x_dim):
                dims, transform, x, y = ("y_1000m", "x_1000m"), transform_1k, x1k, y1k
            elif (h, w) == (y_dim*2, x_dim*2):
                dims, transform, x, y = ("y_500m", "x_500m"), transform_500, x500, y500
            else:
                dims = ("y", "x")
                transform = rasterio.transform.from_bounds(west, south, east, north, h, w)
                x, y = _coords_from_transform(transform, w, h)

            da = xr.DataArray(
                arr,
                dims=dims,
                coords={dims[1]: x, dims[0]: y},
                attrs={
                    "long_name": varname,
                    "scale_factor": scale,
                    "add_offset": offset,
                    "valid_range": (vmin, vmax),
                    "_FillValue": fillvalue,
                },
            )
            da.rio.write_crs(crs, inplace=True)
            da.rio.write_transform(transform, inplace=True)
            data_vars[varname] = da

        file.visititems(collect)

    return xr.Dataset(data_vars)


def load_viirs_collection(paths):
    datasets = []
    for p in paths:
        ds = load_viirs_as_xarray(p)
        t = get_datetime_from_filename(p)
        # add time coordinate to all variables
        ds = ds.expand_dims(time=[t])
        datasets.append(ds)
    return xr.merge(datasets)  #, fill_value=32767)

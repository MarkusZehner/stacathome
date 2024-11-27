import geopandas as gpd
import numpy as np
import pandas as pd
from itertools import product

from dask import delayed
from odc.stac import load as odc_load
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros

from rasterio.errors import WarpOperationError

@delayed
def delayed_store_chunks_to_zarr(
    dataset,
    zarr_store,
    b,
    t_index_slices,
    yx_index_slices,
    chunksize_xy,
    x,
    y,
    dimension_names,
):
    """
    delayed wrapper for store_chunks_to_zarr
    """
    return store_chunks_to_zarr(
        dataset,
        zarr_store,
        b,
        t_index_slices,
        yx_index_slices,
        chunksize_xy,
        x,
        y,
        dimension_names=dimension_names,
    )


def store_chunks_to_zarr(
    dataset,
    zarr_store,
    b,
    t_index_slices,
    yx_index_slices,
    chunksize_xy,
    x,
    y,
    dimension_names=None,
):
    """
    Store chunks to a zarr store.
    Uses region to store only the chunk.
    assumes x and y are the start of the chunk, and chunksize_xy is the chunk size.


    Parameters
    ----------
    dataset: xarray.Dataset
        The dataset to store.
    zarr_store: str
        The path to the zarr store.
    b: str
        The band to store.
    t_index_slices: tuple
        The time index slices.
    yx_index_slices: list
        The yx index slices.
    chunksize_xy: int
        The chunk size.
    x: int
        The x index.
    y: int
        The y index.
    dimension_names: dict
        The dimension names. default is {'time':'time', 'latitude':'latitude', 'longitude':'longitude'}.
        for UTM crs use {'time':'time', 'latitude':'y', 'longitude':'x'}.
    """
    (
        dataset[b]
        .isel(
            {
                dimension_names["longitude"]: slice(x, x + chunksize_xy),
                dimension_names["latitude"]: slice(y, y + chunksize_xy),
            }
        )
        .to_dataset(name=b)
        .to_zarr(
            zarr_store,
            mode="r+",
            write_empty_chunks=False,
            region={
                dimension_names["time"]: slice(*t_index_slices),
                dimension_names["latitude"]: slice(
                    yx_index_slices[0] + y, yx_index_slices[0] + y + chunksize_xy
                ),
                dimension_names["longitude"]: slice(
                    yx_index_slices[2] + x, yx_index_slices[2] + x + chunksize_xy
                ),
            },
        )
    )




def write_otf_subset(
    subset_slices_id_items, tmin, zarr_path, req_chunking, dimension_names
):
    subset, slices, chunk_id, items = subset_slices_id_items

    # items = gdf.clip(subset.boundingbox)['asset_items'].to_list()
    # local_gdf = gdf.copy()
    if len(items) == 0:
        # local_gdf.loc[chunk_id, 'timerange_in_zarr'].append(['No items in subset'])
        return chunk_id, 0, 0, ["No items in subset"]
    # items = gdf.clip(subset.boundingbox)['asset_items'].compute().to_list()
    # if len(items) == 0:
    #    return chunk_id, None, None, 'No items in subset'
    cube = _load_otf_cube_bulk(
        subset=subset, filtered_items=items, requested_bands=None, chunking=req_chunking
    )

    min_time = cube["time"].min().values
    max_time = cube["time"].max().values

    # return chunk_id, min_time, max_time, 'done'

    cube = cube.reindex(
        time=pd.date_range(min_time, max_time, freq="1D"), fill_value=0, method=None
    ).chunk(req_chunking)

    t_insert_start = (
        (min_time - pd.to_datetime(tmin).to_numpy())
        .astype("timedelta64[D]")
        .astype(int)
    )
    t_insert_end = (max_time - pd.to_datetime(tmin).to_numpy()).astype(
        "timedelta64[D]"
    ).astype(int) + 1

    try:
        cube.to_zarr(
            zarr_path,
            mode="r+",
            write_empty_chunks=False,
            region={
                dimension_names["time"]: slice(t_insert_start, t_insert_end),
                dimension_names["latitude"]: slice(slices[0], slices[1]),
                dimension_names["longitude"]: slice(slices[2], slices[3]),
            },
        )

    # except CPLE_AppDefinedError as e:
    #     #return chunk_id, min_time, max_time, f'Error writing to zarr {e}'
    #     #local_gdf.loc[chunk_id, 'timerange_in_zarr'].append([f'{e}'])
    #     #return chunk_id, 0, 0, f'{e}'
    #     return None
    except (WarpOperationError, Exception) as e:
        # return chunk_id, min_time, max_time, f'Error writing to zarr {e}'
        # local_gdf.loc[chunk_id, 'timerange_in_zarr'].append([f'{e}'])
        # return chunk_id, 0, 0, f'{e}'
        return chunk_id, 0, 0, e
        # print(e)

    # return chunk_id, min_time, max_time
    # local_gdf.loc[chunk_id, 'timerange_in_zarr'].append([np.datetime_as_string(min_time, unit='D'),
    #                                                     np.datetime_as_string(max_time, unit='D')])
    # return (chunk_id, np.datetime_as_string(min_time, unit='D'),
    #         np.datetime_as_string(max_time, unit='D'), 'sucess')
    # TODO: write back status and saved time to chunk_table
    return chunk_id, min_time, max_time, "sucess"

def _load_otf_cube_bulk(subset, filtered_items, requested_bands=None, chunking=None):
    """
    Load items on the fly into a cube.

    provides a cube with the requested bands.
    If subset is a chunk id (int) or lat_lon(tuple) chunked according to the chunking, or to a provided geobox.

    Parameters
    ----------
    subset: GeoBox
        The geobox to subset the data to.
    filtered_items: list of Items
        filtered items to load.

    Returns
    -------
    otf_cube: xarray.Dataset
        The loaded cube.
    """
    if requested_bands is None:
        requested_bands = ["B02", "B03", "B04", "B8A"]

    avail_bands = list(set([k for i in filtered_items for k in i.assets.keys()]))
    [a for a in avail_bands if a in requested_bands]

    if chunking is None:
        chunking = {"time": 1000, "latitude": 256, "longitude": 256}

    otf_cube = odc_load(
        filtered_items,
        bands=avail_bands,
        chunks=chunking,
        geobox=subset,
        dtype="uint16",
        resampling="bilinear",
        groupby="solar_day",
    )
    otf_cube = otf_cube.drop_vars(["spatial_ref"])
    otf_cube["time"] = otf_cube.time.dt.floor("D")

    return otf_cube


def get_slice_from_large_data(
    dataset, lat_slice, lon_slice, time_slice=None, dimension_names=None
):
    if not dimension_names:
        dimension_names = {
            "time": "time",
            "latitude": "latitude",
            "longitude": "longitude",
        }

    if time_slice:
        return dataset.isel(
            {
                dimension_names["time"]: slice(*time_slice),
                dimension_names["latitude"]: slice(*lat_slice),
                dimension_names["longitude"]: slice(*lon_slice),
            }
        )
    else:
        return dataset.isel(
            {
                dimension_names["latitude"]: slice(*lat_slice),
                dimension_names["longitude"]: slice(*lon_slice),
            }
        )


def chunk_table_from_geobox(geobox, chunksize_xy, crs=4326, aoi=None):
    num_lon_chunks = geobox.shape.y // chunksize_xy
    num_lat_chunks = geobox.shape.x // chunksize_xy

    lon_ch_id = list(
        range(chunksize_xy // 2, num_lon_chunks * chunksize_xy, chunksize_xy)
    )
    lat_ch_id = list(
        range(chunksize_xy // 2, num_lat_chunks * chunksize_xy, chunksize_xy)
    )

    lon_lat_chunks = list(product(lon_ch_id, lat_ch_id))
    lon_lat_coords = [__lat_lon_from_geobox(geobox, *i) for i in lon_lat_chunks]

    df_locs = pd.DataFrame(
        {
            "lon_chunk": [l[0] for l in lon_lat_chunks],
            "lat_chunk": [l[1] for l in lon_lat_chunks],
            "lon_coord": [l[0] for l in lon_lat_coords],
            "lat_coord": [l[1] for l in lon_lat_coords],
        }
    )

    gpd_locs = gpd.GeoDataFrame(
        df_locs,
        geometry=gpd.points_from_xy(df_locs.lon_coord, df_locs.lat_coord),
        crs=f"EPSG:{crs}",
    )
    if aoi:
        gpd_locs = gpd_locs.clip(aoi)
        gpd_locs = gpd_locs.sort_values(["lon_chunk", "lat_chunk"]).reset_index(
            drop=True
        )
        gpd_locs["chunk_id"] = gpd_locs.index
    return gpd_locs


def __lat_lon_from_geobox(geobox, x, y):
    a = geobox[x, y].affine
    return (a[2], a[5])



def get_geobox_from_aoi(aoi, epsg, resolution, chunksize):
    # buffer to adjust to chunksize
    bounds = aoi.buffer(resolution * chunksize / 2).bounds
    geobox = GeoBox.from_bbox(bounds, crs=f"epsg:{epsg}", resolution=resolution)

    # slice to be a multiple of chunksize
    geobox = geobox[
        slice(
            geobox.shape[0] % chunksize // 2,
            -(chunksize - geobox.shape[0] % chunksize // 2),
        ),
        slice(
            geobox.shape[1] % chunksize // 2,
            -(chunksize - geobox.shape[1] % chunksize // 2),
        ),
    ]
    return geobox


def make_valid_slices_within_geobox(geobox, index_slices):
    if index_slices[0] < 0:
        index_slices[0] = 0
    if index_slices[1] > geobox.shape[0]:
        index_slices[1] = geobox.shape[0]
    if index_slices[2] < 0:
        index_slices[2] = 0
    if index_slices[3] > geobox.shape[1]:
        index_slices[3] = geobox.shape[1]
    return index_slices


def subset_geobox_by_chunk_nr(
    geobox, chunk_nr, chunksize_xy, chunk_table, enlarge_by_n_chunks=0
):
    chunksize_buffered = chunksize_xy + chunksize_xy * enlarge_by_n_chunks * 2
    chunk = chunk_table.loc[chunk_nr]
    ch_side = chunksize_buffered // 2
    index_slices = [
        chunk["lon_chunk"] - ch_side,
        chunk["lon_chunk"] + ch_side,
        chunk["lat_chunk"] - ch_side,
        chunk["lat_chunk"] + ch_side,
    ]

    index_slices = make_valid_slices_within_geobox(geobox, index_slices)

    assert (index_slices[1] - index_slices[0]) % chunksize_xy == 0
    assert (index_slices[3] - index_slices[2]) % chunksize_xy == 0

    return (
        geobox[index_slices[0] : index_slices[1], index_slices[2] : index_slices[3]],
        index_slices,
    )


def get_index_of_chunk_by_latlon(lat, lon, chunk_table):
    return np.sqrt(
        (chunk_table["lat_coord"] - lat) ** 2 + (chunk_table["lon_coord"] - lon) ** 2
    ).argmin()


def subset_geobox_by_chunks_latlon(
    geobox, lat, lon, chunksize_xy, chunk_table, enlarge_by_n_chunks=0
):
    index_ = get_index_of_chunk_by_latlon(lat, lon, chunk_table)
    print(index_)

    return subset_geobox_by_chunk_nr(
        geobox, index_, chunksize_xy, chunk_table, enlarge_by_n_chunks
    )


def create_skeleton_zarr(
    geobox,
    zarr_path,
    chunking=None,
    dtype=np.uint16,
    bands=None,
    start_date=None,
    end_date=None,
    t_freq="1D",
    overwrite=False,
):
    """
    Create a zarr store with the same structure as the datacube, but with no data.
    Time dimension is created with the given start_date, end_date and t_freq.
    This should accomodate differing timestamps in the cube chunks.

    Intended for S-2 data currently.

    Parameters
    ----------
    geobox: GeoBox
        The geobox of the data
    zarr_path: str
        The path to the zarr store
    chunking: tuple
        The chunking of the data, dimension order are 'time', 'latitude/y', 'longitude/x'
    dtype: numpy.dtype
        The datatype of the data
    bands: list
        The bands to include in the zarr store
    start_date: str
        The start date of the time dimension
    end_date: str
        The end date of the time dimension
    t_freq: str
        The frequency of the time dimension

    """
    if not bands:
        bands = ["B02", "B03", "B04", "B8A"]
    assert set(bands) <= set(
        [
            "AOT",
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B09",
            "B11",
            "B12",
            "B8A",
            "SCL",
            "WVP",
        ]
    )

    if chunking is None:
        chunking = (1000, 256, 256)

    xr_empty = xr_zeros(
        geobox,
        dtype=dtype,
        chunks=chunking,
        time=pd.date_range(start_date, end_date, freq=t_freq),
    )
    xr_empty = xr_empty.to_dataset(name=bands[0])
    for band in bands:
        xr_empty[band] = xr_empty[bands[0]].astype(dtype)
        xr_empty[band].encoding["_FillValue"] = 0
    mode = "w" if overwrite else "w-"
    xr_empty.to_zarr(zarr_path, mode=mode, write_empty_chunks=False, compute=False)
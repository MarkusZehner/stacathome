from os import path as os_path, remove as os_remove
import pickle
from copy import copy

import numpy as np
from tqdm import tqdm
from functools import partial

import pandas as pd
import geopandas as gpd
from json import load as json_load
from zarr.errors import ContainsGroupError
from xarray import open_zarr as xr_open_zarr

from shapely import from_geojson, transform
from shapely.geometry import box as s_box, Polygon

import dask
import dask.bag
import dask.backends

from odc.stac import configure_rio, load as odc_load
from pystac_client import Client as pystacClient
from pystac import ItemCollection, Item
from odc.geo.geobox import GeoBox

from .utils import get_transform, get_countries_json, remove_empty_folders
from .request import parallel_request
from .download import parallel_download
from .cuber import (get_geobox_from_aoi, chunk_table_from_geobox, 
                    subset_geobox_by_chunk_nr, subset_geobox_by_chunks_latlon, 
                    create_skeleton_zarr, write_otf_subset,
                    delayed_store_chunks_to_zarr, store_chunks_to_zarr,
                    get_slice_from_large_data)
from .stac import (items_to_dataframe, check_request_against_local, get_all_local_assets,
                   get_unique_elements, check_parallel_request, check_assets)
from .plot import leaflet_overview
from .__version import __version__

configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})


def load_maxicube(path):
    with open(path, "rb") as f:
        mxc = pickle.load(f)
        if not isinstance(mxc, MaxiCube):
            raise ValueError("File is not a MaxiCube")
        if mxc._version != __version__:
            raise ValueError("Version mismatch")
        return mxc


class MaxiCube:
    """
    Class to manage large data cubes from STAC data.
    Most functions are designed to be used with minimal additional input arguments.

    Will create a geobox on initialization, and a chunk table to manage the data,
    and checks local assets of the given path.
    Locally available items are stored in self.items_local_global.

    If a zarr_path is provided, an empty zarr store will be created with the correct structure.

    Parameters
    ----------
    aoi: str
        The area of interest, either a country name or file path to a geojson.
    path: str
        The path to the folder for this MaxiCube.
    crs: int, default 4326
        The EPSG code of the coordinate reference system.
    resolution: float, default 0.00018
        The resolution of the data in units of the crs.
    chunksize_xy: int, default 256
        The size of the x and y chunks.
    chunksize_t: int, default -1
        The size of the time chunks.
    url: str, default 'https://planetarycomputer.microsoft.com/api/stac/v1'
        The url of the STAC API.
    collection: str, default 'sentinel-2-l2a'
        The collection to search for.
    requested_bands: list, default ['B02', 'B03', 'B04', 'B8A']
        The bands to request from the STAC API.
    save_name: str (optional)
        The name of the pickle file to save the local items.
    zarr_path: str (optional)
        The path to the zarr store.
    """

    def __init__(
        self,
        aoi: str,
        path: str,
        crs: int = 4326,
        resolution: float = 0.00018,
        chunksize_xy: int = 256,
        chunksize_t: int = -1,
        url: str = None,
        collection: str = "sentinel-2-l2a",
        requested_bands: list[str] = None,
        zarr_path: str = None,
    ):
        # manage data structure
        self._version = __version__
        self.name = aoi
        self.crs = crs
        self.transform = None
        if self.crs != 4326:
            self.transform = get_transform(self.crs, 4326)
            self.dimension_names = {"time": "time", "latitude": "y", "longitude": "x"}
        else:
            self.dimension_names = {
                "time": "time",
                "latitude": "latitude",
                "longitude": "longitude",
            }

        self.aoi = self.get_aoi(aoi)

        self.resolution = resolution
        self.chunksize_xy = chunksize_xy
        self.chunksize_t = chunksize_t
        self.geobox = get_geobox_from_aoi(
            self.aoi, self.crs, self.resolution, self.chunksize_xy
        )
        self.chunk_table = chunk_table_from_geobox(
            self.geobox, self.chunksize_xy, self.crs, self.aoi
        )
        self.chunk_table["timerange_in_zarr"] = np.empty(
            (len(self.chunk_table), 0)
        ).tolist()
        if not requested_bands:
            self.requested_bands = ["B02", "B03", "B04", "B8A"]
        else:
            self.requested_bands = requested_bands

        # manage data assets
        if not url:
            self.url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        else:
            self.url = url
        self.collection = collection

        self.path = path
        if self.path:
            self.items_local_global = self.local_assets()
        else:
            self.items_local_global = None
        self.req_items = None
        self.pending = None
        self.req_items_local = None

        if zarr_path:
            self.zarr_path = zarr_path
            self.construct_large_cube()
        else:
            self.zarr_path = None

    def __str__(self):
        return f"MaxiCube {self.name} with {len(self.items_local_global)} items"
    
    def __repr__(self):
        return f"MaxiCube {self.name} with {len(self.items_local_global)} items"

    def get_aoi(self, aoi: str):
        if isinstance(aoi, str):
            if isinstance(aoi, str) and os_path.exists(aoi):
                with open(aoi) as f:
                    json = json_load(f)
            else:
                json = get_countries_json(aoi)

            json_crs = None
            if (
                "properties" in json["features"][0]
                and "crs" in json["features"][0]["properties"]
            ):
                json_crs = json["features"][0]["properties"]["crs"]

            aoi = from_geojson(f"{json['features'][0]['geometry']}".replace("'", '"'))

            if json_crs is not None and json_crs != self.crs:
                aoi = transform(aoi, get_transform(json_crs, self.crs), include_z=False)
            return aoi
        
        elif isinstance(aoi, Polygon):
            return aoi
        
        elif isinstance(aoi, tuple):
            return s_box(*aoi)
        
        else:
            raise ValueError("Invalid aoi")

    def items_as_geodataframe(self, items: list[Item] = None):
        """
        Return the items as a GeoDataFrame.
        """
        if items is None:
            items = self.items_local_global
        return items_to_dataframe(items)

    def plot_chunk_table(self):
        """
        Plot the chunk centroids from the chunk table as a map.
        """
        return self.chunk_table.plot(
            column="chunk_id", cmap="viridis", legend=True, markersize=0.05
        )

    def subset(
        self,
        chunk_id: int = None,
        lat_lon: tuple[float] = None,
        enlarge_by_n_chunks: int = 0,
    ):
        """
        Subset the geobox by chunk_id or lat_lon.
        Will adhere to the chunks.

        Parameters
        ----------
        chunk_id: int
            The chunk id to subset by.
        lat_lon: tuple
            The lat lon to which teh nearest chunk is returned.
        enlarge_by_n_chunks: int
            The number of chunks to enlarge the subset by, 1 results in a 3x3 chunk subset.

        Returns
        -------
        subset: GeoBox
            The subsetted geobox.
        yx_index_slices: list
            The indecex for slices to subset the data within larger zarr store.
        """
        if chunk_id is not None:
            subset, yx_index_slices = subset_geobox_by_chunk_nr(
                self.geobox,
                chunk_id,
                self.chunksize_xy,
                self.chunk_table,
                enlarge_by_n_chunks,
            )
        elif lat_lon is not None:
            subset, yx_index_slices = subset_geobox_by_chunks_latlon(
                self.geobox,
                lat_lon[0],
                lat_lon[1],
                self.chunksize_xy,
                self.chunk_table,
                enlarge_by_n_chunks,
            )
        else:
            raise ValueError("Must provide either chunk_id or lat_lon")
        return subset, yx_index_slices

    # def _parse_subset(self, subset:int|tuple[float]|GeoBox, enlarge_by_n_chunks=0):
    #     if isinstance(subset, int):
    #         subset, yx_index_slices = self.subset(
    #             chunk_id=subset, enlarge_by_n_chunks=enlarge_by_n_chunks)
    #     elif isinstance(subset, tuple):
    #         subset, yx_index_slices = self.subset(
    #             lat_lon=subset, enlarge_by_n_chunks=enlarge_by_n_chunks)
    #     elif isinstance(subset, GeoBox):
    #         subset = subset
    #     return subset, index_slices

    def request_items(
        self,
        start_date: str,
        end_date: str,
        subset: int | tuple[float] | GeoBox,
        enlarge_by_n_chunks=0,
        collection=None,
        url=None,
        new_request=False,
    ):
        """
        Request items from the STAC API.

        Writes the requested items to self.req_items, and compares them to the local assets.
        Available items are written to self.req_items_local, and pending items to self.pending.

        Parameters
        ----------
        start_date: str
            The start date of the request.
        end_date: str
            The end date of the request.
        subset: int, tuple or None
            The subset to request items for.
        enlarge_by_n_chunks: int
            The number of chunks to enlarge the subset by.
        collection: str
            The collection to request items from.
        url: str
            The url of the STAC API.
        new_request: bool
            If True, will request new items, if False will use the last request.
        """
        # TODO: Subset parser?
        if subset is None and subset == "full":
            subset = self.aoi
            print("Using full aoi this may take a while")
        elif isinstance(subset, int):
            subset, _ = self.subset(
                chunk_id=subset, enlarge_by_n_chunks=enlarge_by_n_chunks
            )
            subset = s_box(*subset.boundingbox)
        elif isinstance(subset, tuple):
            subset, _ = self.subset(
                lat_lon=subset, enlarge_by_n_chunks=enlarge_by_n_chunks
            )
            subset = s_box(*subset.boundingbox)

        if not collection:
            collection = self.collection
        else:
            raise UserWarning(
                "Using collection defined in function call - may differ from init"
            )
        if not url:
            url = self.url
        else:
            raise UserWarning(
                "Using url defined in function call - may differ from init"
            )

        if self.transform is not None:
            subset = transform(subset, self.transform, include_z=False)

        catalog = pystacClient.open(url)
        found = catalog.search(
            bbox=subset.boundary.bounds,
            datetime=f"{start_date}/{end_date}",
            collections=[collection],
        ).item_collection()
        if self.req_items is None or new_request == True:
            self.req_items = list(found)
        else:
            self.__extend_request_items(found)
        self.compare_local(report=True)

    def __extend_request_items(self, items):
        ids = [i.id for i in self.req_items]
        for i in items:
            if i.id not in ids:
                self.req_items.append(i)

    def compare_local(self, report=False):
        if not self.req_items:
            raise ValueError("Request items first")
        self.req_items_local, self.pending = check_request_against_local(
            self.req_items,
            self.path,
            requested_bands=self.requested_bands,
            report=report,
        )

    def local_assets(self, name="stacathome_local_items.pkl", rerequest=False):
        """
        Load local assets from the pickle file, or request them if not available.

        Parameters
        ----------
        rerequest: bool


        Returns
        -------
        items_local_global: list
            The local items.
        """
        if not rerequest and os_path.exists(os_path.join(self.path, name)):
            print("Loading local assets")
            return pickle.load(open(os_path.join(self.path, name), "rb"))
        else:
            print("Requesting local assets")
            return get_all_local_assets(
                self.path, self.collection, self.requested_bands
            )

    def plot(self, items=None, plot_chunks=True, subset_chunks_by=50):
        """
        Plot the items on a map.

        Parameters
        ----------
        items: list
            The items to plot.
        plot_chunks: bool
            If True, will plot the chunk centroids.
        subset_chunks_by: int
            The number of every nth chunk to plot for faster plotting.

        Returns
        -------
        m: folium.Map
            The map.
        """
        # ENH: add option to plot requested/avail items
        if not items:
            items = self.items_local_global

        chunks = None
        if plot_chunks:
            chunks = self.chunk_table.iloc[::subset_chunks_by]

        gdf = items_to_dataframe(items)
        return leaflet_overview(gdf, chunks, aoi=self.aoi, transform_to=self.transform)

    def download_requested(self, items=None, use_dask=True, daskkwargs={}):
        print("deprecated, should be done with download_all")
        # """
        # Download the items.
        # Will use self.req_items if no items are provided.

        # Parameters
        # ----------
        # items: list
        #     The items to download.
        # use_dask: bool
        #     If True, will use dask to parallelize the download.
        # daskkwargs: dict
        #     The dask arguments.

        # Returns
        # -------
        # str
        #     A message that all items are downloaded.
        # """
        # if not items:
        #     if not self.req_items:
        #         raise ValueError(
        #             'Request items first, or provide items to the download function')
        #     items = self.req_items
        # else:
        #     raise UserWarning(
        #         'Using items defined in function call - may differ from request')

        # self.compare_local(report=False)

        # if len(self.pending) == 0:
        #     print('All requested items already downloaded')
        #     return

        # if use_dask:
        #     print('deprecated')
        #     # # Create a SLURM cluster
        #     # cluster = SLURMCluster(
        #     #     queue=daskkwargs['queue'] if 'queue' in daskkwargs else 'work',
        #     #     cores=daskkwargs['cores'] if 'cores' in daskkwargs else 1,
        #     #     processes=daskkwargs['processes'] if 'processes' in daskkwargs else 1,
        #     #     memory=daskkwargs['memory'] if 'memory' in daskkwargs else '500MB',
        #     #     walltime=daskkwargs['walltime'] if 'walltime' in daskkwargs else '03:00:00',
        #     # )

        #     # if 'min_workers' in daskkwargs and 'max_workers' in daskkwargs:
        #     #     cluster.adapt(
        #     #         minimum=daskkwargs['min_workers'], maximum=daskkwargs['max_workers'])
        #     # elif 'num_workers' in daskkwargs:
        #     #     cluster.scale(jobs=daskkwargs['num_workers'])
        #     # else:
        #     #     cluster.adapt(minimum=1, maximum=20)

        #     # # Create a Dask client that connects to the cluster
        #     # client = daskClient(cluster)

        #     # # Check cluster status
        #     # print(cluster)

        #     # downloads = db.from_sequence(self.pending).map(download_item)
        #     # downloads.compute()

        #     # client.close()
        #     # cluster.close()
        # else:
        #     for i in tqdm(self.pending):
        #         get_asset(*i)

        # self._update_items_local_global(items)

    def merge_items(self, items=None):
        """
        Merge the items with self.items_local_global.
        defaults to self.req_items_local, the requested, locally available items.

        Parameters
        ----------
        items: list
            The items to merge.
        """

        if not items:
            if not self.req_items_local:
                print("Request and download items first!")
                return
            items = self.req_items_local

        new_ids = [i.id for i in items]
        old_ids = [i.id for i in self.items_local_global]

        for i, id_ in enumerate(new_ids):
            if id_ not in old_ids:
                self.items_local_global.append(items[i])
            else:
                idx = old_ids.index(id_)
                for band in items[i].assets:
                    if band not in self.items_local_global[idx].assets:
                        self.items_local_global[idx].assets[band] = items[i].assets[
                            band
                        ]

    def save_items(self, items=None, name="stacathome_local_items.pkl"):
        """
        Save the items to a pickle file.
        defaults to self.req_items, compares local assets first, then saves all local items.


        Parameters
        ----------
        items: list
            The items to save.
        """

        self._update_items_local_global(items)
        with open(os_path.join(self.path, name), "wb") as f:
            pickle.dump(self.items_local_global, f)

    def status(self):
        print(
            (
                f"Items requested: {len(self.req_items) if self.req_items else 0}, Items requestedlocal : {len(
            self.req_items_local) if self.req_items_local else 0}, Items local: {len(self.items_local_global) if self.items_local_global else 0}"
            )
        )

    def _update_items_local_global(self, items=None):
        if items is None and self.req_items is None:
            self.req_items = copy(self.items_local_global)
        elif items is not None and type(items[0]) == ItemCollection:
            self.req_items = get_unique_elements(items)

        elif items is not None and type(items[0]) == Item:
            self.req_items = items
        self.compare_local(report=False)
        self.merge_items()
        self.req_items = None
        self.req_items_local = None
        self.pending = None
        self.items_local_global = [
            i for i in self.items_local_global if len(i.assets.keys()) > 0
        ]
        print(f"Updated local items, {len(self.items_local_global)} items")

    def save(self, name="saved.maxicube"):
        """
        Save the MaxiCube Object to a pickle file.
        """
        self._update_items_local_global()
        with open(os_path.join(self.path, f"{name}"), "wb") as f:
            pickle.dump(self, f)

    def download_all(
        self,
        start_date,
        end_date,
        client=None,
        subset=None,
        enlarge_by_n_chunks=0,
        grid_size=None,
        subdivision=1,
        verbose=False,
    ):
        """
        Intended wrapper to request, download and store items in one go.
        Will request items in parallel, check them against local assets, 
        download them in parallel and store them.
        Uses the given AOI and chunk table to request items for a regular grid of points.
        """
        # TODO: Subset parser?
        if subset is None:
            subset = self.aoi
            print("Using full aoi this may take a while")
        elif isinstance(subset, int):
            subset, _ = self.subset(
                chunk_id=subset, enlarge_by_n_chunks=enlarge_by_n_chunks
            )
            subset = s_box(*subset.boundingbox)
        elif isinstance(subset, tuple):
            print("subset by lat lon")
            subset, _ = self.subset(
                lat_lon=subset, enlarge_by_n_chunks=enlarge_by_n_chunks
            )
            subset = s_box(*subset.boundingbox)
        elif isinstance(subset, GeoBox):
            subset = s_box(*subset.boundingbox)
        elif isinstance(subset, Polygon):
            subset = subset
        process = parallel_request(
            start_date,
            end_date,
            self.collection,
            self.url,
            self.transform,
            subset,
            self.crs,
            grid_size,
        )
        
        checked = check_parallel_request(process, self.requested_bands, self.path)
        if len(checked) == 0:
            if verbose:
                print("No new items to download")
            return process
        if subdivision > 1:
            checked = [
                checked[i : i + subdivision]
                for i in range(0, len(checked), subdivision)
            ]
        parallel_download(checked, client)
        # self._update_items_local_global(process) # does not make sense here with fire_and_forget download
        return process

    def load_otf_cube(
        self,
        subset,
        items=None,
        requested_bands=None,
        chunking=None,
        enlarge_by_n_chunks=0,
        drop_fill=False,
    ):
        # TODO remake with bulk functions
        """
        Load items on the fly into a cube.

        provides a cube with the requested bands.
        If subset is a chunk id (int) or lat_lon(tuple) chunked according to the chunking, or to a provided geobox.

        Parameters
        ----------
        items: list
            The items to load.
        subset: int, tuple or GeoBox
            The geobox to subset the data to.
        requested_bands: list
            The bands to load.
        chunking: dict
            The chunking of the data.

        Returns
        -------
        otf_cube: xarray.Dataset
            The loaded cube.
        """

        if requested_bands is None:
            requested_bands = self.requested_bands
        if chunking is None:
            chunking = {
                self.dimension_names["time"]: self.chunksize_t,
                self.dimension_names["latitude"]: self.chunksize_xy,
                self.dimension_names["longitude"]: self.chunksize_xy,
            }

        # TODO: Subset parser?
        if isinstance(subset, int):
            subset, _ = self.subset(
                chunk_id=subset, enlarge_by_n_chunks=enlarge_by_n_chunks
            )
        elif isinstance(subset, tuple):
            subset, _ = self.subset(
                lat_lon=subset, enlarge_by_n_chunks=enlarge_by_n_chunks
            )
        elif isinstance(subset, GeoBox):
            subset = subset

        if items is None:
            items = self.items_local_global
        gdf = items_to_dataframe(items, self.crs)
        items = gdf.clip(subset.boundingbox)["asset_items"].to_list()

        if len(items) == 0:
            raise ValueError("No items in subset")

        otf_cube = odc_load(
            items,
            bands=requested_bands,
            chunks=chunking,
            geobox=subset,
            dtype="uint16",
            resampling="bilinear",
            groupby="solar_day",
        )
        if drop_fill:
            mean_over_time = otf_cube[requested_bands[0]].mean(
                dim=[
                    self.dimension_names["latitude"],
                    self.dimension_names["longitude"],
                ]
            )
            otf_cube = otf_cube.isel(time=np.where(mean_over_time != 0)[0])
        return otf_cube

    def construct_large_cube(
        self,
        zarr_path=None,
        start_date="2015-06-01",
        end_date="2026-01-01",
        chunking=None,
        overwrite=False,
    ):
        """
        This cube is created empty, start and end date should contain entire planned time series.

        Parameters
        ----------
        zarr_path: str
            The path to the zarr store.
        start_date: str
            The start date of the time dimension.
        end_date: str
            The end date of the time dimension.
        chunking: dict
            The chunking of the data, dimension names are 'time', 'latitude', 'longitude' for crs 4326, 'time', 'y', 'x' for utm.
        overwrite: bool
            If True, will overwrite the zarr store.
        """
        if zarr_path is None:
            if self.zarr_path is None:
                raise ValueError(
                    "No zarr path provided, set self.zarr_path or provide it to the function"
                )
            zarr_path = self.zarr_path
        else:
            self.zarr_path = zarr_path

        if chunking is None:
            chunking = {
                self.dimension_names["time"]: self.chunksize_t,
                self.dimension_names["latitude"]: self.chunksize_xy,
                self.dimension_names["longitude"]: self.chunksize_xy,
            }
        try:
            create_skeleton_zarr(
                self.geobox,
                zarr_path,
                chunking=(
                    chunking[self.dimension_names["time"]],
                    chunking[self.dimension_names["latitude"]],
                    chunking[self.dimension_names["longitude"]],
                ),
                dtype=np.uint16,
                bands=self.requested_bands,
                start_date=start_date,
                end_date=end_date,
                t_freq="1D",
                overwrite=overwrite,
            )
        except ContainsGroupError as e:
            print(
                f"Zarr already exists at {
                  zarr_path}. Skipping creation. Set overwrite=True to overwrite."
            )

    def check_assets_for_read_errors(self, subdivision=20):
        self._update_items_local_global()
        sliced_items = [
            self.items_local_global[i : i + subdivision]
            for i in range(0, len(self.items_local_global), subdivision)
        ]
        res_2 = []
        for i in tqdm(sliced_items):
            res_2.append(dask.delayed(check_assets)(i))
        res_2 = dask.compute(*res_2)

        not_found, read_failed = [], []
        for r in res_2:
            if len(r[0]) > 0:
                not_found.extend(r[0])
            if len(r[1]) > 0:
                read_failed.extend(r[1])

        for i in not_found:
            # this should not happen
            if os_path.exists(i):
                print(i, os_path.getsize(i) // 1000000)
            else:
                print("not exist:", i)

        ids = [i.id for i in self.items_local_global]
        empty_items = []
        for i in read_failed:
            if os_path.exists(i):
                path_sep = i.split("/")
                item_name = path_sep[-6]
                item_name = item_name[:27] + item_name[33:-5]
                asset = path_sep[-1][-11:-8]

                try:
                    pos = ids.index(item_name)
                    if asset in self.items_local_global[pos].assets:
                        del self.items_local_global[pos].assets[asset]
                        if len(self.items_local_global[pos].assets) == 0:
                            empty_items.append(pos)
                except ValueError:
                    continue
                print(
                    "removed ",
                    item_name,
                    asset,
                    " with size ",
                    os_path.getsize(i) // 1000000,
                )
                os_remove(i)
            else:
                print("not exist:", i)
        for i in empty_items[::-1]:
            del self.items_local_global[i]

    def fill_large_cube(
        self, subset=None, enlarge_by_n_chunks=0, items=None, client=None, dask=False
    ):
        # TODO simplify/cleanup, handle all similar to _fill_all_large_data with write_otf_subset
        """
        Fill the large cube with the requested items.


        Parameters
        ----------
        subset: int, tuple or list
            The subset to fill.
        enlarge_by_n_chunks: int
            The number of chunks to enlarge the subset by.
        items: list
            The items to fill the cube with.
        dask: bool
            If True, will return dask tasks.

        Returns
        -------
        tasks: list
            The dask tasks.
        """

        if self.zarr_path is None:
            raise ValueError(
                "No zarr path provided, set self.zarr_path or run construct_large_cube first"
            )
        else:
            self.construct_large_cube()
            large_cube = xr_open_zarr(self.zarr_path)
            t_min_large_cube = large_cube.time.min().values

        if items is None:
            items = self.items_local_global

        if isinstance(subset, int) or isinstance(subset, tuple):
            tasks = self.__fill_subset_into_large_data(
                subset, items, t_min_large_cube, enlarge_by_n_chunks, dask
            )
        elif isinstance(subset, list):
            tasks = []
            for s in subset:
                tasks.extend(
                    self.__fill_subset_into_large_data(
                        s, items, t_min_large_cube, enlarge_by_n_chunks, dask
                    )
                )
        elif subset is None:
            print("Filling entire cube, this may take a while")
            res = self._fill_all_large_data(client)
            return res

        if dask:
            return tasks

    def _fill_all_large_data(self, client, subdivision=1):
        """
        Fill the entire cube with the local items using given chunking.
        """
        # gdf = dg.from_geopandas(self.items_as_geodataframe(), npartitions=20)
        # client.scatter(self.items_as_geodataframe())
        items_gdf = self.items_as_geodataframe()
        req_chunking = {
            self.dimension_names["time"]: -1,
            self.dimension_names["latitude"]: self.chunksize_xy,
            self.dimension_names["longitude"]: self.chunksize_xy,
        }

        t_min_large_cube = xr_open_zarr(self.zarr_path).time.min().values

        print("prepare subsets")
        partial_load_otf_subset = partial(
            write_otf_subset,
            tmin=t_min_large_cube,
            zarr_path=self.zarr_path,
            req_chunking=req_chunking,
            dimension_names=self.dimension_names,
        )
        res = []
        for i in tqdm(range(0, len(self.chunk_table), 100)):
            futures = []
            for j in range(i, i + 100):
                subset, slice_ = self.subset(j)
                items = items_gdf.clip(subset.boundingbox)["asset_items"].to_list()
                futures.append(
                    dask.delayed(partial_load_otf_subset)((subset, slice_, j, items))
                )
            res_loop = dask.compute(*futures)

            for j in res_loop:
                self.chunk_table.loc[j[0], "timerange_in_zarr"].append([j[1], j[2]])
            res.extend(res_loop)
        return res

    def __fill_subset_into_large_data(
        self, subset, items, t_min_large_cube, enlarge_by_n_chunks, dask=False
    ):
        """
        Fill the large cube with the requested items.

        Parameters
        ----------
        subset: int, tuple or list
            The subset to fill.
        items: list
            The items to fill the cube with.
        t_min_large_cube: str
            The minimum time of the large cube.
        enlarge_by_n_chunks: int
            The number of chunks to enlarge the subset by.
        dask: bool
            If True, will return dask tasks.

        Returns
        -------
        tasks: list
            The dask tasks.
        """
        tasks = []

        # TODO: Subset parser?
        if subset is None:
            subset = self.aoi
        elif isinstance(subset, int):
            subset, yx_index_slices = self.subset(
                chunk_id=subset, enlarge_by_n_chunks=enlarge_by_n_chunks
            )
        elif isinstance(subset, tuple):
            subset, yx_index_slices = self.subset(
                lat_lon=subset, enlarge_by_n_chunks=enlarge_by_n_chunks
            )

        req_chunking = {
            self.dimension_names["time"]: -1,
            self.dimension_names["latitude"]: self.chunksize_xy,
            self.dimension_names["longitude"]: self.chunksize_xy,
        }

        try:
            otf_cube = self.load_otf_cube(subset, items, chunking=req_chunking)
        except ValueError as e:
            print(e)
            return

        otf_cube = otf_cube.drop_vars(["spatial_ref"])
        otf_cube["time"] = otf_cube.time.dt.floor("D")

        min_time = otf_cube["time"].min().values
        max_time = otf_cube["time"].max().values

        # add time range saved of chunk in chunk table
        self.chunk_table.loc[self.chunk_table.clip(subset.boundingbox).index][
            "timerange_in_zarr"
        ] = self.chunk_table.loc[self.chunk_table.clip(subset.boundingbox).index][
            "timerange_in_zarr"
        ].apply(
            lambda x: x.append([min_time, max_time])
        )

        t_insert_start = (
            (min_time - pd.to_datetime(t_min_large_cube).to_numpy())
            .astype("timedelta64[D]")
            .astype(int)
        )
        t_insert_end = (max_time - pd.to_datetime(t_min_large_cube).to_numpy()).astype(
            "timedelta64[D]"
        ).astype(int) + 1

        otf_cube = otf_cube.reindex(
            time=pd.date_range(min_time, max_time, freq="1D"), fill_value=0, method=None
        ).chunk(req_chunking)

        for b in self.requested_bands:
            for x in range(
                0, len(otf_cube[self.dimension_names["longitude"]]), self.chunksize_xy
            ):
                for y in range(
                    0,
                    len(otf_cube[self.dimension_names["latitude"]]),
                    self.chunksize_xy,
                ):
                    if dask:
                        tasks.append(
                            delayed_store_chunks_to_zarr(
                                otf_cube,
                                self.zarr_path,
                                b,
                                (t_insert_start, t_insert_end),
                                yx_index_slices,
                                self.chunksize_xy,
                                x,
                                y,
                                self.dimension_names,
                            )
                        )
                    else:
                        store_chunks_to_zarr(
                            otf_cube,
                            self.zarr_path,
                            b,
                            (t_insert_start, t_insert_end),
                            yx_index_slices,
                            self.chunksize_xy,
                            x,
                            y,
                            self.dimension_names,
                        )
        if dask:
            return tasks

    def get_chunk(self, chunk_id, time_slice=None, drop_fill=False):
        """
        Get a chunk from the large cube, na are filled with 0 to save space.

        Parameters
        ----------
        chunk_id: int or tuple
            The chunk id or lat_lon to get.
        time_slice: tuple
            The time slice to get.
        drop_fill: bool
            If True, will drop all-fill value time slices.

        Returns
        -------
        chunk_data: xarray.Dataset
            The chunk data.
        """
        if self.zarr_path:
            # TODO: Subset parser?
            if isinstance(chunk_id, int):
                _, yx_index_slices = self.subset(chunk_id=chunk_id)
            elif isinstance(chunk_id, tuple):
                _, yx_index_slices = self.subset(lat_lon=chunk_id)

            if time_slice is not None:
                if isinstance(time_slice[0], str):
                    large_cube = xr_open_zarr(self.zarr_path)
                    t_min_large_cube = large_cube.time.min().values
                    t_slice_start = (
                        (
                            pd.to_datetime(time_slice[0]).to_numpy()
                            - pd.to_datetime(t_min_large_cube).to_numpy()
                        )
                        .astype("timedelta64[D]")
                        .astype(int)
                    )
                    t_slice_end = (
                        (
                            pd.to_datetime(time_slice[1]).to_numpy()
                            - pd.to_datetime(t_min_large_cube).to_numpy()
                        )
                        .astype("timedelta64[D]")
                        .astype(int)
                    )
                time_slice = (t_slice_start, t_slice_end + 1)
            chunk_data = get_slice_from_large_data(
                xr_open_zarr(self.zarr_path, mask_and_scale=False),
                lat_slice=yx_index_slices[:2],
                lon_slice=yx_index_slices[2:],
                time_slice=time_slice,
                dimension_names=self.dimension_names,
            )
            if drop_fill:
                mean_over_time = chunk_data[self.requested_bands[0]].mean(
                    dim=[
                        self.dimension_names["latitude"],
                        self.dimension_names["longitude"],
                    ]
                )
                chunk_data = chunk_data.isel(time=np.where(mean_over_time != 0)[0])
            return chunk_data

    def remove_tile(self, tile, delete_assets=False):
        """
        Delete all data from one tile in the dataset.

        """
        to_remove = []
        hrefs = []
        for i, a in enumerate(self.items_local_global):
            if a.properties["s2:mgrs_tile"] == tile:
                to_remove.append(i)
                if delete_assets:
                    for assets in a.assets:
                        try:
                            os_remove(a.assets[assets].href)
                            hrefs.append(os_path.dirname(a.assets[assets].href))
                        except Exception as e:
                            print(e)

        for i in to_remove[::-1]:
            del self.items_local_global[i]

        if delete_assets:
            remove_empty_folders(set(hrefs), self.path)
        return self.status()

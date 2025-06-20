import os
import logging
from pathlib import Path

from stacathome.registry import PROCESSOR_REGISTRY, get_supported_bands, get_processor
from stacathome.generic_utils import parse_time, most_common, cube_to_zarr_zip


class STACRequest():
    def __init__(
        self,
        collections: list[str] | str,
        request_place: any,
        request_time: any,
        request_bands=None,
    ):
        AVAIL_COLLECTIONS = set(PROCESSOR_REGISTRY.keys())
        self.unmatched_collections = set()

        if isinstance(collections, str):
            collections = [collections]

        self.collections = []
        for c in collections:
            if c in AVAIL_COLLECTIONS:
                self.collections.append(c)
            else:
                self.unmatched_collections.add(c)
        if self.unmatched_collections:
            logging.warning(f"Some collections were not matched: {self.unmatched_collections}")

        self.request_place = request_place
        if isinstance(request_time, dict):
            self.request_time = {k: parse_time(v) for k, v in request_time.items()}
        else:
            self.request_time = parse_time(request_time)

        self.request_bands = {}
        if request_bands:
            if isinstance(request_bands, dict):
                request_bands = [item for sublist in request_bands.values() for item in sublist]
            for c in self.collections:
                self.request_bands[c] = list(set(get_supported_bands(c)) & set(request_bands))
                if len(self.request_bands[c]) == 0:
                    self.request_bands[c] = get_supported_bands(c)
        else:
            for c in self.collections:
                self.request_bands[c] = get_supported_bands(c)

        # will get a list of sensors, lat long or shape or bbox and time
        # finds the needed tiles for a given request by probing the STAC
        # should nudge the provided location towards grid matching the data sources if not specified otherwise
        # gathers all items and returns them with geoboxes per gridding in the sensors

    # refactor to:
    # request items: filter the items to most prevalent utm tile if reduce overlap is set to true in the processor

    def __split_req_time(self, collection):
        if isinstance(self.request_time, dict):
            req_time = self.request_time.get(collection, None)
            if req_time is None:
                logging.warning(f"No time found for {collection} in {self.request_place}")
        else:
            req_time = self.request_time
        return req_time

    def request_items(self, **kwargs):
        items_per_collection = {}
        for collection in self.collections:
            req_time = self.__split_req_time(collection)
            if req_time is None:
                continue

            items = get_processor(collection).request_items(collection, req_time, self.request_place, **kwargs)

            logging.info(f"Found {len(items)} items for {collection}")
            items_per_collection[collection] = items
        return items_per_collection

    def filter_items(self, found_items):
        returned_items_per_collection = {}
        for collection in self.collections:
            items = found_items[collection]
            if len(items) == 0:
                logging.info(f'no items found for {collection}')
                continue
            if get_processor(collection).gridded:
                if get_processor(collection).overlap:
                    items = get_processor(items[0]).collect_covering_tiles_and_coverage(self.request_place, items=items)[0]
                else:
                    items = [i for i in items if get_processor(i).does_cover_data(self.request_place, input_crs=4326)]
            else:
                items = [i for i in items]

            logging.info(f"{len(items)} items for {collection} after filter")
            returned_items_per_collection[collection] = items
        return returned_items_per_collection

    def download_tiles(self, path: Path, items, collections=None):
        collections = self.collections if collections is None else collections
        collections = [collections] if isinstance(collections, str) else collections
        ret_paths = {}
        for collection in collections:
            item = items[collection]
            bands = set(get_supported_bands(collection)) & set(self.request_bands[collection])

            collection_path = path / collection
            os.makedirs(collection_path, exist_ok=True)
            ret_paths[collection] = get_processor(collection).download_tiles_to_file(collection_path, item, bands)
        return ret_paths

    def create_stac_items(self, paths, collections=None):
        collections = self.collections if collections is None else collections
        collections = [collections] if isinstance(collections, str) else collections
        ret_items = {}
        for collection in collections:
            col_path = paths.get(collection, None)
            if col_path is None:
                logging.warning(f"No paths found for {collection}, skipping STAC item generation")
                continue
            ret_items[collection] = get_processor(collection).generate_stac_items(col_path)
        return ret_items

    def create_geoboxes(self, items: dict):
        tile_ids_per_collection = {}
        for collection in self.collections:
            collection_items = items[collection]
            if len(collection_items) == 0:
                logging.warning(f"No items in {collection}")
                continue
            proc = get_processor(collection_items[0])
            if not proc.gridded:
                logging.warning(f"Collection {collection} is not gridded, skipping geobox creation")
                continue

            geobox = proc.get_geobox(self.request_place)

            tile_ids_per_collection[collection] = geobox
        return tile_ids_per_collection

    def get_data(self, path: Path = None, chunks: dict = None, name_ident=None):

        if path is not None and path is not isinstance(path, Path):
            path = Path(path)
        # load data: load data to file, and to cube
        # depending on the processors setting:
        # cube (native per odc, preferred via pystac, userdefined if further steps are required e.g S3)
        items = self.request_items()
        items = self.filter_items(items)

        gboxes = {}
        cubes = {}
        for collection in self.collections:
            proc_cls = get_processor(collection)

            paths = None
            if proc_cls.cubing in ['preferred', 'custom']:
                paths = self.download_tiles(path, items, collection)

            if proc_cls.cubing in ['preferred']:
                items[collection] = self.create_stac_items(paths, collection)[collection]

            # return items

            if proc_cls.cubing in ['native', 'preferred']:
                most_common_crs = most_common([get_processor(i).get_crs() for i in items[collection]])

                most_common_crs_item = next(
                    i
                    for i in items[collection]
                    if get_processor(i).get_crs() == most_common_crs)

                gboxes[collection] = get_processor(most_common_crs_item).get_geobox(self.request_place)

                # cube  (native, preferred)
                for key, cube in self.load_cubes_basic(items,
                                                       gboxes,
                                                       subset_bands_per_collection=self.request_bands,
                                                       path=path,
                                                       chunks=chunks,
                                                       name_ident=name_ident,
                                                       collections=collection).items():
                    cubes[key] = cube

            if proc_cls.cubing in ['preferred']:
                # self.tidy_up()
                pass

        return cubes

    def load_cubes_basic(
            self,
            items: dict,
            geoboxes: dict,
            subset_bands_per_collection: dict = None,
            path: Path = None,
            savefunc: callable = cube_to_zarr_zip,
            split_by: int = None,
            chunks: dict = None,
            collections: list[str] | None = None,
            name_ident=None,
    ):
        return_cubes = {}

        collections = self.collections if collections is None else collections
        collections = [collections] if isinstance(collections, str) else collections

        for collection in collections:
            item = items[collection]
            logging.info(f"Loading {len(item)} items for {collection} as cube")
            bands = get_supported_bands(collection)
            subset_bands = subset_bands_per_collection.get(collection, None) if subset_bands_per_collection else self.request_bands.get(collection, None)
            bands = [band for band in bands if band in subset_bands]
            if len(bands) == 0:
                logging.warning(f"No bands found for {collection} in {self.request_place} and {self.request_time}")
                continue

            geobox = geoboxes.get(collection, None)
            if geobox is None:
                logging.warning(f"No geobox found for {collection} in {self.request_place} and {self.request_time}")
                continue

            proc = get_processor(item[0])
            logging.debug(f"{len(item)} items for {collection} after filtering")
            data = proc.load_cube(item, bands, geobox, split_by, chunks)

            if isinstance(data, dict):
                for platform, dat in data.items():
                    if path and savefunc:
                        if isinstance(path, str):
                            path = Path(path)
                        os.makedirs(path, exist_ok=True)
                        if name_ident:
                            collection_name = f"{platform}_{name_ident}"
                        else:
                            collection_name = platform
                        savefunc(path / (collection_name + '.zarr.zip'), dat)

                    return_cubes[platform] = dat

            else:
                if path and savefunc:
                    if isinstance(path, str):
                        path = Path(path)
                    os.makedirs(path, exist_ok=True)
                    if name_ident:
                        collection_name = f"{collection}_{name_ident}"
                    else:
                        collection_name = collection
                    savefunc(path / (collection_name + '.zarr.zip'), data)

                return_cubes[collection] = data
        return return_cubes
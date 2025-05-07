import os
import logging
from collections import defaultdict
from pathlib import Path

from shapely import box
from odc.geo.geobox import GeoBox

# from stacathome.utils import run_with_multiprocessing_and_return
from stacathome.redo_classes.registry import PROCESSOR_REGISTRY, get_supported_bands, get_tilename_key, get_processor
from stacathome.redo_classes.providers import STACProvider
from stacathome.redo_classes.generic_utils import parse_time, most_common, resolve_best_containing, merge_to_cover, cube_to_zarr_zip


logging.basicConfig(
    level=logging.INFO,  # Set to WARNING or ERROR in production
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class STACRequest():
    def __init__(
        self,
        collections: list[str] | str,
        request_place: any,
        request_time: any
    ):
        AVAIL_COLLECTIONS = set(PROCESSOR_REGISTRY.keys())
        # add other provider collection here
        self.stac_providers = defaultdict(list)
        unmatched_collections = set()

        if isinstance(collections, str):
            collections = [collections]
        for c in collections:
            if c in AVAIL_COLLECTIONS:
                self.stac_providers[STACProvider()].append(c)
            # add other providers here
            else:
                unmatched_collections.add(c)
        if unmatched_collections:
            logging.warning(f"Some collections were not matched: {unmatched_collections}")

        self.request_place = request_place
        if isinstance(request_time, dict):
            self.request_time = {k: parse_time(v) for k, v in request_time.items()}
        else:
            self.request_time = parse_time(request_time)

        # will get a list of sensors, lat long or shape or bbox and time
        # finds the needed tiles for a given request by probing the STAC
        # should nudge the provided location towards grid matching the data sources if not specified otherwise
        # gathers all items and returns them with geoboxes per gridding in the sensors

    def collect_covering_tiles_and_coverage(self, item_limit=12):
        tile_ids_per_collection = {}
        for provider, collections in self.stac_providers.items():
            for collection in collections:
                if isinstance(self.request_time, dict):
                    req_time = self.request_time.get(collection, None)
                    if req_time is None:
                        logging.warning(f"No time found for {collection} in {self.request_place}")
                        continue
                else:
                    req_time = self.request_time
                print(req_time, flush=True)
                items = provider.request_items(collection=collection,
                                               request_time=req_time,
                                               request_place=self.request_place,
                                               max_items=item_limit,
                                               max_retry=5)
                if len(items) < item_limit:
                    logging.warning(f"Less than {item_limit} items found for {collection} in {self.request_place} "
                                    f"and {self.request_time}")

                by_tile = defaultdict(list)

                for i in items:
                    item = get_processor(i)
                    by_tile[item.get_tilename_value()].append([
                        item.get_crs(),
                        item.contains_shape(self.request_place),
                        item.centroid_distance_to(self.request_place),
                        item.overlap_percentage(self.request_place),
                        item.get_bbox(),
                    ])

                # Reduce each group using majority voting
                by_tile_filtered = [
                    [tile_id] + [most_common(attr) for attr in zip(*vals)]
                    for tile_id, vals in by_tile.items()
                ]

                # First, try finding a containing item
                best = resolve_best_containing(by_tile_filtered)
                if best:
                    found_tiles = [best]
                else:
                    found_tiles = merge_to_cover(by_tile_filtered, self.request_place)

                tile_ids = [t[0] for t in found_tiles]

                # get one item with the same tile_id for the geobox
                item = next(get_processor(i)
                            for i in items
                            if get_processor(i).get_tilename_value() == tile_ids[0])

                geobox = item.get_geobox(self.request_place)

                tile_ids_per_collection[collection] = {'tile_id': tile_ids, 'geobox': geobox}
        return tile_ids_per_collection

    def request_items_basic(self, **kwargs):
        returned_items_per_collection = {}
        for provider, collections in self.stac_providers.items():
            for collection in collections:
                if isinstance(self.request_time, dict):
                    req_time = self.request_time.get(collection, None)
                    if req_time is None:
                        logging.warning(f"No time found for {collection} in {self.request_place}")
                        continue
                else:
                    req_time = self.request_time
                items = provider.request_items(collection=collection,
                                               request_time=req_time,
                                               request_place=self.request_place,
                                               **kwargs)
                if get_processor(items[0]).gridded:
                    items = [i for i in items if get_processor(i).does_cover_data(self.request_place, input_crs=4326)]
                else:
                    items = [i for i in items]

                logging.info(f"Found {len(items)} items for {collection}")
                returned_items_per_collection[collection] = items
        return returned_items_per_collection

    def create_geoboxes(self, items: dict, item_limit=12):
        tile_ids_per_collection = {}
        for provider, collections in self.stac_providers.items():
            for collection in collections:
                collection_items = items[collection]
                if len(collection_items) == 0:
                    logging.warning(f"No items found for {collection} in {self.request_place} and {self.request_time}")
                    continue
                proc = get_processor(collection_items[0])
                if not proc.gridded:
                    logging.warning(f"Collection {collection} is not gridded, skipping geobox creation")
                    continue

                # if proc.tilename is None:
                #     logging.warning(f"Collection {collection} has no tilename, skipping geobox creation")
                #     continue

                geobox = proc.get_geobox(self.request_place)

                tile_ids_per_collection[collection] = geobox
        return tile_ids_per_collection

    def request_items(self, tile_ids: dict = None, geobox: GeoBox = None, **kwargs):
        returned_items_per_collection = {}
        for provider, collections in self.stac_providers.items():
            for collection in collections:
                if isinstance(self.request_time, dict):
                    req_time = self.request_time.get(collection, None)
                    if req_time is None:
                        logging.warning(f"No time found for {collection} in {self.request_place}")
                        continue
                else:
                    req_time = self.request_time
                tile_id = tile_ids[collection]['tile_id']
                filter_arg = get_tilename_key(collection)
                items = provider.request_items(collection=collection,
                                               request_time=req_time,
                                               query={filter_arg: {'in': tile_id}},
                                               **kwargs)

                logging.info(f"Found {len(items)} items for {collection}")
                returned_items_per_collection[collection] = items
        return returned_items_per_collection

    def load_granules(self, items: dict, path: Path = None):
        loaded_dict = {}
        for provider, collections in self.stac_providers.items():
            prov_item_list = []
            for collection in collections:
                item = items[collection]
                bands = get_supported_bands(collection)
                for i in item:
                    assets = i.get_assets()
                    selected_assets = [assets.get(band, None).href for band in bands]
                    if path:
                        selected_assets = [(s, path / s.split('/')[-1]) for s in selected_assets]
                    prov_item_list.extend(selected_assets)

                loaded_dict[collection] = prov_item_list
            if path:
                provider.download_granules_to_file(prov_item_list)

            # handle load granules to memory?
            # return iterator that loads the tiles?

        return loaded_dict

    def load_cubes(
            self,
            items: dict,
            geoboxes: dict,
            subset_bands_per_collection: dict = None,
            path: Path = None,
            savefunc: callable = cube_to_zarr_zip,
            split_by: int = None,
            chunks: dict = None
    ):
        if isinstance(path, str):
            path = Path(path)

        os.makedirs(path, exist_ok=True)

        return_cubes = {}
        for provider, collections in self.stac_providers.items():
            for collection in collections:
                item = items[collection]
                logging.info(f"Loading {len(item)} items for {collection} as cube")
                bands = get_supported_bands(collection)
                subset_bands = subset_bands_per_collection.get(collection, None)
                if subset_bands:
                    bands = [band for band in bands if band in subset_bands]
                    if len(bands) == 0:
                        logging.warning(f"No bands found for {collection} in {self.request_place} and {self.request_time}")
                        continue

                geobox = geoboxes[collection]['geobox']

                proc = get_processor(item[0])
                # if collection == 'modis-13Q1-061':
                #     return geobox, proc, item
                item = [i for i in item if get_processor(i).does_cover_data(box(*list(geobox.keys())[0].boundingbox))]
                item = proc.sort_items_by_datetime(items=item)
                # data = run_with_multiprocessing_and_return(
                #     proc.load_cube,
                #     items=item,
                #     bands=bands,
                #     geobox=geobox,
                #     provider=provider,
                #     path=path,
                #     savefunc=savefunc
                # )
                logging.debug(f"{len(item)} items for {collection} after filtering")
                data = proc.load_cube(item, bands, geobox, provider, split_by, chunks)

                if path and savefunc:
                    savefunc(path / (collection + '.zarr.zip'), data)

                return_cubes[collection] = data
        return return_cubes

    def load_cubes_basic(
            self,
            items: dict,
            geoboxes: dict,
            subset_bands_per_collection: dict = None,
            path: Path = None,
            savefunc: callable = cube_to_zarr_zip,
            split_by: int = None,
            chunks: dict = None
    ):
        return_cubes = {}
        for provider, collections in self.stac_providers.items():
            for collection in collections:
                item = items[collection]
                logging.info(f"Loading {len(item)} items for {collection} as cube")
                bands = get_supported_bands(collection)

                if subset_bands_per_collection:
                    subset_bands = subset_bands_per_collection.get(collection, [])
                    bands = [band for band in bands if band in subset_bands]
                    if len(bands) == 0:
                        logging.warning(f"No bands found for {collection} in {self.request_place} and {self.request_time}")
                        continue

                geobox = geoboxes.get(collection, None)
                if geobox is None:
                    logging.warning(f"No geobox found for {collection} in {self.request_place} and {self.request_time}")
                    continue

                proc = get_processor(item[0])
                # if collection == 'modis-13Q1-061':
                #     return geobox, proc, item
                # item = [i for i in item if get_processor(i).does_cover_data(box(*list(geobox.keys())[0].boundingbox))]
                item = proc.sort_items_by_datetime(items=item)
                # data = run_with_multiprocessing_and_return(
                #     proc.load_cube,
                #     items=item,
                #     bands=bands,
                #     geobox=geobox,
                #     provider=provider,
                #     path=path,
                #     savefunc=savefunc
                # )
                logging.debug(f"{len(item)} items for {collection} after filtering")
                data = proc.load_cube(item, bands, geobox, provider, split_by, chunks)

                if path and savefunc:
                    if isinstance(path, str):
                        path = Path(path)
                    os.makedirs(path, exist_ok=True)
                    savefunc(path / (collection + '.zarr.zip'), data)

                return_cubes[collection] = data
        return return_cubes

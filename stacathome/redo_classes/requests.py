import os
import logging
from collections import defaultdict
from pathlib import Path

from shapely import box
# from odc.geo.geobox import GeoBox

# from stacathome.utils import run_with_multiprocessing_and_return
from stacathome.redo_classes.registry import PROCESSOR_REGISTRY, get_supported_bands, get_processor, request_items, request_items_tile, get_tilename_key
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
        request_time: any,
        request_bands: None,
    ):
        AVAIL_COLLECTIONS = set(PROCESSOR_REGISTRY.keys())
        # AVAIL_COLLECTIONS_STAC = set(PROCESSOR_REGISTRY_STAC.keys())
        # AVAIL_COLLECTIONS_ASF = set(PROCESSOR_REGISTRY_ASF.keys())
        # add other provider collection here
        # self.stac_providers = defaultdict(list)
        self.unmatched_collections = set()

        # stac_provider = STACProvider()
        # asf_provider = ASFProvider()

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

            # l_items = len(items)
            # if get_processor(collection).overlap:
            #     items = self.collect_covering_tiles_and_coverage(items=items)
            #     l_items = len(items[0])

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

            # if proc.tilename is None:
            #     logging.warning(f"Collection {collection} has no tilename, skipping geobox creation")
            #     continue

            geobox = proc.get_geobox(self.request_place)

            tile_ids_per_collection[collection] = geobox
        return tile_ids_per_collection

    # def tidy_up(self):
    #     pass

    def get_data(self, path: Path, chunks: dict = None, name_ident=None):

        if not isinstance(path, Path):
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

            if proc_cls.cubing in ['preferred', 'custom']:
                paths = self.download_tiles(path, items, collection)

            if proc_cls.cubing in ['preferred']:
                items[collection] = self.create_stac_items(paths, collection)[collection]

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
                # cubes[collection] = self.load_cubes_basic(items,
                #                                           gboxes,
                #                                           subset_bands_per_collection=self.request_bands,
                #                                           path=path,
                #                                           chunks=chunks,
                #                                           name_ident=name_ident,
                #                                           collections=collection)[collection]

            if proc_cls.cubing in ['preferred']:
                # self.tidy_up()
                pass

        return cubes

#     def collect_covering_tiles_and_coverage(self, item_limit=12, items=None):
#         raise DeprecationWarning(
#             "collect_covering_tiles_and_coverage in REQUESTS is deprecated, use processor.collect_covering_tiles_and_coverage instead.")
#         # move this into the processor classes?
#         tile_ids_per_collection = {}
#         for collection in self.collections:
#             if isinstance(self.request_time, dict):
#                 req_time = self.request_time.get(collection, None)
#                 if req_time is None:
#                     logging.warning(f"No time found for {collection} in {self.request_place}")
#                     continue
#             else:
#                 req_time = self.request_time

#             if items is None:
#                 items = get_processor(collection).request_items(
#                     request_time=req_time,
#                     request_place=self.request_place,
#                     max_items=item_limit)
#             else:
#                 filter = True

#             if len(items) < item_limit:
#                 logging.warning(f"Less than {item_limit} items found for {collection} in {self.request_place} "
#                                 f"and {self.request_time}")

#             collect_coverage_from = items[:min(len(items), item_limit)]

#             by_tile = defaultdict(list)

#             for i in collect_coverage_from:
#                 item = get_processor(i)
#                 by_tile[item.get_tilename_value()].append([
#                     item.get_crs(),
#                     item.contains_shape(self.request_place),
#                     item.centroid_distance_to(self.request_place),
#                     item.overlap_percentage(self.request_place),
#                     item.get_bbox(),
#                 ])

#             # Reduce each group using majority voting
#             by_tile_filtered = [
#                 [tile_id] + [most_common(attr) for attr in zip(*vals)]
#                 for tile_id, vals in by_tile.items()
#             ]

#             # First, try finding a containing item
#             best = resolve_best_containing(by_tile_filtered)
#             if best:
#                 found_tiles = [best]
#             else:
#                 found_tiles = merge_to_cover(by_tile_filtered, self.request_place)

#             tile_ids = [t[0] for t in found_tiles]

#             # get one item with the same tile_id for the geobox
#             item = next(get_processor(i)
#                         for i in collect_coverage_from
#                         if get_processor(i).get_tilename_value() == tile_ids[0])

#             geobox = item.get_geobox(self.request_place)

#             if filter:
#                 filtered_items = [i for i in items if get_processor(i).get_tilename_value() in tile_ids]

#             tile_ids_per_collection[collection] = {'tile_id': tile_ids, 'geobox': geobox}
#         if filter:
#             return [filtered_items, tile_ids_per_collection]
#         else:
#             return tile_ids_per_collection

#     def request_items_basic(self, **kwargs):
#         returned_items_per_collection = {}
#         for collection in self.collections:
#             if isinstance(self.request_time, dict):
#                 req_time = self.request_time.get(collection, None)
#                 if req_time is None:
#                     logging.warning(f"No time found for {collection} in {self.request_place}")
#                     continue
#             else:
#                 req_time = self.request_time

#             # refactor using the processors instead of providers:
#             items = request_items(collection, req_time, self.request_place, **kwargs)
#             # items = provider.request_items(collection=collection,
#             #                                request_time=req_time,
#             #                                request_place=self.request_place,
#             #                                **kwargs)
#             if len(items) == 0:
#                 logging.info(f'no items found for {collection}')
#                 continue
#             if get_processor(items[0]).gridded:
#                 items = [i for i in items if get_processor(i).does_cover_data(self.request_place, input_crs=4326)]
#             else:
#                 items = [i for i in items]

#             logging.info(f"Found {len(items)} items for {collection}")
#             returned_items_per_collection[collection] = items
#         return returned_items_per_collection

#     def request_items_old(self, tile_ids: dict = None, **kwargs):
#         returned_items_per_collection = {}
#         for collection in self.collections:
#             if isinstance(self.request_time, dict):
#                 req_time = self.request_time.get(collection, None)
#                 if req_time is None:
#                     logging.warning(f"No time found for {collection} in {self.request_place}")
#                     continue
#                 else:
#                     req_time = self.request_time
#                 tile_id = tile_ids[collection]['tile_id']

#                 if get_tilename_key(collection) is not None:
#                     items = request_items_tile(collection, req_time, tile_id, **kwargs)
#                 else:
#                     logging.info(f"Collection {collection} has no tile id, skipping tile request")

#                 # items = provider.request_items(collection=collection,
#                 #                                request_time=req_time,
#                 #                                query={filter_arg: {'in': tile_id}},
#                 #                                **kwargs)

#                 logging.info(f"Found {len(items)} items for {collection}")
#                 returned_items_per_collection[collection] = items
#         return returned_items_per_collection

#     def load_granules(self, items: dict, path: Path = None):
#         loaded_dict = {}
#         for collection in self.collections:
#             item = items[collection]
#             bands = get_supported_bands(collection)
#             for i in item:
#                 assets = i.get_assets()
#                 selected_assets = [assets.get(band, None).href for band in bands]
#                 if path:
#                     selected_assets = [(s, path / s.split('/')[-1]) for s in selected_assets]

#             loaded_dict[collection] = selected_assets
#             if path:
#                 get_processor(item[0]).download_granules_to_file(selected_assets)

#             # handle load granules to memory?
#             # return iterator that loads the tiles?

#         return loaded_dict

#     def load_cubes(
#             self,
#             items: dict,
#             geoboxes: dict,
#             subset_bands_per_collection: dict = None,
#             path: Path = None,
#             savefunc: callable = cube_to_zarr_zip,
#             split_by: int = None,
#             chunks: dict = None
#     ):
#         if isinstance(path, str):
#             path = Path(path)

#         os.makedirs(path, exist_ok=True)

#         return_cubes = {}
#         for collection in self.collections:
#             item = items[collection]
#             logging.info(f"Loading {len(item)} items for {collection} as cube")
#             bands = get_supported_bands(collection)
#             subset_bands = subset_bands_per_collection.get(collection, None) if subset_bands_per_collection else self.request_bands.get(collection, None)
#             if subset_bands:
#                 bands = [band for band in bands if band in subset_bands]
#                 if len(bands) == 0:
#                     logging.warning(f"No bands found for {collection} in {self.request_place} and {self.request_time}")
#                     continue

#             geobox = geoboxes[collection]['geobox']

#             proc = get_processor(item[0])
#             # if collection == 'modis-13Q1-061':
#             #     return geobox, proc, item
#             item = [i for i in item if get_processor(i).does_cover_data(box(*list(geobox.keys())[0].boundingbox))]
#             item = proc.sort_items_by_datetime(items=item)
#             # data = run_with_multiprocessing_and_return(
#             #     proc.load_cube,
#             #     items=item,
#             #     bands=bands,
#             #     geobox=geobox,
#             #     provider=provider,
#             #     path=path,
#             #     savefunc=savefunc
#             # )
#             logging.debug(f"{len(item)} items for {collection} after filtering")
#             data = proc.load_cube(item, bands, geobox, split_by, chunks)

#             if path and savefunc:
#                 savefunc(path / (collection + '.zarr.zip'), data)

#             return_cubes[collection] = data
#         return return_cubes

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
            # if collection == 'modis-13Q1-061':
            #     return geobox, proc, item
            # item = [i for i in item if get_processor(i).does_cover_data(box(*list(geobox.keys())[0].boundingbox))]
            # item = proc.sort_items_by_datetime(items=item)
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


def stacathome_wrapper(area, time_range, collections, subset_bands, path_out, name_ident):
    request = STACRequest(collections, area, time_range, subset_bands)
    return request.get_data(path=path_out, name_ident=name_ident)

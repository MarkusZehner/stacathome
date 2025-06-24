import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Iterable

from stacathome.registry import PROCESSOR_REGISTRY, get_supported_bands, get_processor
from stacathome.generic_utils import most_common


class STACRequest():
    def __init__(
        self,
        collection: str,
        location: str,
        start_time: datetime,
        end_time: datetime,
        variables: Iterable[str] | None = None,
    ):
        if collection not in PROCESSOR_REGISTRY.keys():
            raise ValueError(f'{collection} is not a valid collection. Available: {list(PROCESSOR_REGISTRY.keys())}')

        if start_time >= end_time:
            raise ValueError('end_time must be after start_time')

        available_vars = get_supported_bands(collection)
        variables = variables or available_vars

        for var in variables:
            if var not in available_vars:
                raise ValueError(f'Variable {var} is not included in collection {collection}. Available: {available_vars}')

        self.collection = collection
        self.location = location
        self.start_time = start_time
        self.end_time = end_time
        self.variables = set(variables)
        self.processor_factory = get_processor(self.collection)        


    def request_items(self, **kwargs):
        items = self.processor_factory.request_items(self.collection, [self.start_time, self.end_time], self.location, **kwargs)
        return items

    def filter_items(self, items):
        if items and self.processor_factory.gridded:
            if self.processor_factory.overlap:
                items = self.processor_factory(items[0]).collect_covering_tiles_and_coverage(self.location, items=items)[0]
            else:
                items = [item for item in items if self.processor_factory(item).does_cover_data(self.location, input_crs=4326)]
        return items

    def create_stac_items(self, path):
        return self.processor_factory.generate_stac_items(path)

    def create_geoboxes(self, items):
        if not items:
            raise ValueError('Need at least one item')
        return get_processor(items[0]).get_geobox(self.location)
        
    def get_data(self, chunks: dict = None, name_ident=None):
        # load data: load data to file, and to cube
        # depending on the processors setting:
        # cube (native per odc, preferred via pystac, userdefined if further steps are required e.g S3)
        items = self.request_items()
        items = self.filter_items(items)

        cubes = {}

        # TODO !!!
        if self.processor_factory.cubing in ['preferred', 'custom']:
            paths = self.download_tiles(paths=None, items=items)  # TODO

        # TODO !!!
        if self.processor_factory.cubing in ['preferred']:
            items = self.create_stac_items(paths=None) 

        if not items:
            return cubes  # TODO: decide how to deal with this edgecase

        if self.processor_factory.cubing in ['native', 'preferred']:
            most_common_crs = most_common([get_processor(item).get_crs() for item in items])
            most_common_crs_item = next(item for item in items if get_processor(item).get_crs() == most_common_crs)
            geobox = get_processor(most_common_crs_item).get_geobox(self.location)

            # cube  (native, preferred)
            cubes = self.load_cubes_basic(
                items,
                geobox,
                chunks=chunks,
            )
    
        return cubes

    def load_cubes_basic(
            self,
            items: dict,
            geobox: dict,
            split_by: int = None,
            chunks: dict = None,
    ):
        return_cubes = {}

        processor = get_processor(items[0])
        data = processor.load_cube(items, self.variables, geobox, split_by, chunks)
        if isinstance(data, dict):
            for platform, dat in data.items():
                return_cubes[platform] = dat
        else:
            return_cubes[self.collection] = data
        return return_cubes
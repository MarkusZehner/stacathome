from collections import namedtuple
from datetime import datetime
import re

import shapely
import pystac
import earthaccess
import earthaccess.results
from earthaccess.search import DataCollections
from odc.stac import load
from odc.geo.geom import Geometry
from rio_stac.stac import bbox_to_geom

from ..generic_utils import get_nested
from ..stac import update_stac_item
from .common import BaseProvider, register_provider

from importlib.resources import files
from pathlib import Path
import csv


def save_list_of_tuples(data, path):
    data = sorted(data, key=lambda x: x[0])
    with open(path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["collection", "version"])  # header
        writer.writerows(data)


def load_list_of_tuples(file_name):
    with open(file_name, "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        data = [tuple(row) for row in reader]
    return data

class EarthAccessProvider(BaseProvider):
    def __init__(self, provider_name: str):
        super().__init__(provider_name)
        earthaccess.login(persist=True)

    @property
    def _available_collections(self) -> list[tuple[str, str]]:
        collections_with_versions = None
        path = files("stacathome.metadata.earthaccess") / "earthaccess_datacollections.csv"
        if path.is_file():
            collections_with_versions = load_list_of_tuples(path)

        nr_of_collections = DataCollections().hits()
        if not collections_with_versions or not nr_of_collections == len(collections_with_versions):
            print('collecting')
            avail_datasets = DataCollections().get_all()
            collections_with_versions = [
                (i['umm']['ShortName'], i['umm']['Version']) for i in avail_datasets
            ]
            save_list_of_tuples(collections_with_versions, path)
                    
        shortname_version = namedtuple('EarthaccessCollections', ['collection', 'version'])
        collections_with_versions = [shortname_version(i[0], i[1]) for i in sorted(set(collections_with_versions))]
        return collections_with_versions

    def get_metadata(self, collection):
        '''
        to be used with collection names from available_collections
        '''
        return earthaccess.search_datasets(short_name=collection)
    
    def available_collections(self) -> list[tuple[str, str]]:
        return [item.collection for item in self._available_collections]

    def has_collection_with_version(self, collection:str, newest_only=True):
        '''
        Returns all matching collections with version number.
        '''
        matches = [item for item in self._available_collections if collection.lower() in item.collection.lower()]
        if not newest_only:
            return matches
        highest = {}
        for m in matches:
            if not m.collection in highest or highest.get(m.collection, ('', None))[0] < m.version:
                highest[m.collection] = (m.version, m)
        return [i[1] for i in highest.values()]
            

    def _request_items(
        self,
        collection: str | tuple,
        starttime: datetime,
        endtime: datetime,
        roi: Geometry = None,
        limit: int = None,
        **kwargs,
    ) -> pystac.ItemCollection:
        # nasa item collections have versions. those are separated by a '.' in other documents after the short_name
        version = kwargs.pop('version', None)

        bounding_box=tuple(roi.boundingbox) if roi else None
        if bounding_box:
            kwargs['bounding_box'] = bounding_box
            
        granules = earthaccess.search_data(
            short_name=collection,
            version=version,
            temporal=(starttime, endtime),
            count=limit if limit else -1,
            **kwargs,
        )
        return self.to_itemcollection(granules)

    def to_itemcollection(self, granules: list[earthaccess.results.DataGranule]) -> pystac.ItemCollection:
        items = [self.create_item(granule) for granule in granules]
        item_collection = pystac.item_collection.ItemCollection(items, clone_items=False)
        
        return item_collection

    def create_item(self, granule: earthaccess.results.DataGranule) -> pystac.Item:
        """
        Create a STAC item from a Granule object.
        Args:
            granule (earthaccess.results.Granule): The granule to convert into a STAC item.
        Returns:
            pystac.Item: The created STAC item.
        """
        item_id = get_nested(granule, ['meta', 'native-id'])

        item_start_datetime = get_nested(granule, ['umm', 'TemporalExtent', 'RangeDateTime', 'BeginningDateTime'])
        item_end_datetime = get_nested(granule, ['umm', 'TemporalExtent', 'RangeDateTime', 'EndingDateTime'])
        if item_start_datetime:
            item_start_datetime = datetime.fromisoformat(item_start_datetime.replace("Z", "+00:00"))
        if item_end_datetime:
            item_end_datetime = datetime.fromisoformat(item_end_datetime.replace("Z", "+00:00"))
        item_datetime = None
        if item_datetime:
            item_datetime = datetime.fromisoformat(item_datetime.replace("Z", "+00:00"))

        if not item_start_datetime or not item_end_datetime:
            print(item_start_datetime, item_end_datetime)
            print('item differs in datetime, implement for: ', get_nested(granule, ['umm', 'TemporalExtent']))
            
        item_bbox = None
        item_geometry=None
        geometry_entry = get_nested(granule, ['umm', 'SpatialExtent', 'HorizontalSpatialDomain', 'Geometry'])
        bounds = geometry_entry.get('BoundingRectangles')
        if bounds:
            bounds = bounds[0]
            xmin = bounds['WestBoundingCoordinate']
            xmax = bounds['EastBoundingCoordinate']
            ymin = bounds['SouthBoundingCoordinate']
            ymax = bounds['NorthBoundingCoordinate']
            item_bbox = [xmin, ymin, xmax, ymax]
            item_geometry = bbox_to_geom(item_bbox)
            
        gpolygon = geometry_entry.get('GPolygons')
        if gpolygon:
            poly = shapely.Polygon([shapely.Point(*i.values()) for i in gpolygon[0]['Boundary']['Points']])
            item_geometry = shapely.geometry.mapping(poly)
            item_bbox = poly.bounds
            
        if not item_bbox and not item_geometry:
            print(f'item has different geometry field: {geometry_entry.keys()}')
            

        assets = {Path(entry).name.replace(item_id, '').lstrip('._').split('.')[0]:
            pystac.Asset(
                href = entry,
                ) for entry in granule.data_links()
                }

        item = pystac.Item(
            id=item_id,
            datetime=item_datetime,
            start_datetime=item_start_datetime,
            end_datetime=item_end_datetime,
            geometry=item_geometry,
            bbox=item_bbox,
            properties={'original_result': granule},  # needed for download? -> could be just href
            assets=assets
        )
        
        item.validate()

        return item

    @staticmethod
    def download_urls(urls, out_dir, threads, **kwargs):
        return earthaccess.download(urls, local_path=out_dir, threads=threads, **kwargs)
        
    def load_granule(self, item: pystac.Item, variables:list[str]|None=None,
                     out_dir:str | None = None, threads:int = 1, **kwargs) -> bytes:
        if variables:
            urls = [asset.href for name, asset in item.get_assets().items() if name in variables]
        else:    
            urls = [asset.href for asset in item.get_assets().values()]
        if not out_dir:
            out_dir = ''

        local_files = self.download_urls(urls, out_dir, threads, **kwargs)
        for i, lf in enumerate(local_files):
            p = Path(lf)
            if not p.is_absolute():
                local_files[i] = str(p.resolve())
        
        item = update_stac_item(item, urls, local_files)
        return item

    def create_cube(self, items, parameters):
        data = load(**parameters)
        if data is None:
            raise ValueError("Failed to create cube")
        return data


register_provider('earthaccess', EarthAccessProvider)

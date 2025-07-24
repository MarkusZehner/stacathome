from collections import namedtuple
from datetime import datetime
import re

import pystac
import earthaccess
import earthaccess.results
from odc.stac import load
from odc.geo.geom import Geometry
from rio_stac.stac import bbox_to_geom

from ..generic_utils import get_nested
from .common import BaseProvider, register_provider


class EarthAccessProvider(BaseProvider):
    def __init__(self, provider_name: str):
        super().__init__(provider_name)
        earthaccess.login(persist=True)

    def available_collections(self) -> list[tuple[str, str]]:
        shortname_version = namedtuple('EarthaccessCollections', ['collection', 'version'])
        cloud_hosted_datasets = earthaccess.search_datasets(cloud_hosted=True)
        not_cloud_hosted_datasets = earthaccess.search_datasets(cloud_hosted=False)
        avail_datasets = cloud_hosted_datasets + not_cloud_hosted_datasets
        collections_with_versions = [
            shortname_version(i['umm']['ShortName'], i['umm']['Version']) for i in avail_datasets
        ]
        return list(set(collections_with_versions))
    
    def has_collection(self, collection):
        return any(item.collection == collection for item in self.available_collections())
    
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

        granules = earthaccess.search_data(
            short_name=collection,
            version=version,
            temporal=(starttime, endtime),
            bounding_box=tuple(roi.boundingbox) if roi else None,
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
            
        
        bounds = get_nested(granule, ['umm', 'SpatialExtent', 'HorizontalSpatialDomain', 'Geometry', 'BoundingRectangles'])[0]
        xmin = bounds['WestBoundingCoordinate']
        xmax = bounds['EastBoundingCoordinate']
        ymin = bounds['SouthBoundingCoordinate']
        ymax = bounds['NorthBoundingCoordinate']
        item_bbox = [xmin, ymin, xmax, ymax]
        item_geometry = bbox_to_geom(item_bbox)

        assets = {re.split(r'_(\d{2})_', entry)[-1].replace('.tif', '') :
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
    def download_granule(granules, local_path, threads, **kwargs):
        earthaccess.download(granules, local_path=local_path, threads=threads, **kwargs)
        # update the stac items after download for precise info?

    def create_cube(self, parameters):
        data = load(**parameters)
        if data is None:
            raise ValueError("Failed to create cube")
        return data


register_provider('earthaccess', EarthAccessProvider)

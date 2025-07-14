from collections import namedtuple
from datetime import datetime

import earthaccess
import odc
import odc.stac
from odc.geo.geom import Geometry
import pystac
import shapely

from .common import BaseProvider, register_provider


class EarthAccessProvider(BaseProvider):
    def __init__(self):
        super().__init__('earthaccess')
        earthaccess.login(persist=True)
        
    def available_collections(self) -> list[tuple[str, str]]:
        shortname_version = namedtuple('EarthaccessCollections', ['ShortName', 'Version'])
        cloud_hosted_datasets = earthaccess.search_datasets(cloud_hosted=True)
        not_cloud_hosted_datasets = earthaccess.search_datasets(cloud_hosted=False)
        avail_datasets = cloud_hosted_datasets + not_cloud_hosted_datasets
        collections_with_versions = [shortname_version(i['umm']['ShortName'],
                                                       i['umm']['Version'])  for i in avail_datasets]
        return list(set(collections_with_versions))

    def _request_items(
        self,
        collection: str,
        starttime: datetime,
        endtime: datetime,
        roi: Geometry = None,
        limit: int = None,
        **kwargs,
    ) -> pystac.ItemCollection:
        # nasa item collections have versions. those are separated by a '.' in the short name in other documents
        # collection, version = collection.split('.', 1) if '.' in collection else (collection, None)
        version = None
        if isinstance(collection, tuple):
            try:
                collection, version = collection.ShortName, collection.Version
            except AttributeError:
                collection, version = collection[0], collection[1]
            finally:
                raise TypeError("Collection must be a string or a named tuple with 'ShortName' and 'Version' attributes.")
            
        granules = earthaccess.search_data(
            short_name=collection,
            version=version,
            temporal=(starttime, endtime),
            bounding_box=tuple(roi.boundingbox) if roi else None,
            count=limit if limit else -1,
            **kwargs,
        )
        return granules
    
    # def _create_item(self, granule: earthaccess.results.DataGranule) -> pystac.Item:
    #     """
    #     Create a STAC item from a Granule object.
    #     Args:
    #         granule (earthaccess.Granule): The granule to convert into a STAC item.
    #     Returns:
    #         pystac.Item: The created STAC item.
    #     """
    #     item = granule.to_stac_item()
    #     item.id = granule.id
    #     item.properties['datetime'] = granule.start_time
    #     item.properties['end_datetime'] = granule.end_time
    #     item.properties['collection'] = granule.short_name
    #     return item

    def download_from_earthaccess(self, granules, local_path, threads, **kwargs):
        earthaccess.download(granules, local_path=local_path, threads=threads, **kwargs)

    def create_cube(self, parameters):
        data = odc.stac.load(**parameters)
        if data is None:
            raise ValueError("Failed to create cube")
        return data


register_provider(EarthAccessProvider)

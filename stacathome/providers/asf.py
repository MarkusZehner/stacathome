from datetime import datetime
from collections import defaultdict
from pathlib import Path

import asf_search
import odc
import odc.stac
import pystac
import shapely

from ..generic_utils import get_nested
from ..stac import update_stac_item
from .common import BaseProvider, register_provider



class ASFProvider(BaseProvider):

    def __init__(self, provider_name: str):
        super().__init__(provider_name)

    def available_collections(self):
        return [i for i in asf_search.CMR.datasets.dataset_collections]

    def _request_items(
        self,
        collection: str,
        starttime: datetime,
        endtime: datetime,
        roi: shapely.Geometry = None,
        limit: int = None,
        **kwargs,
    ) -> pystac.ItemCollection:
        wkt = roi.wkt
        results = asf_search.search(
            dataset=collection,
            start=starttime,
            end=endtime,
            intersectsWith=wkt,
            maxResults=limit,
            **kwargs,
        )

        return self.to_itemcollection(results)
    
    def to_itemcollection(self, granules: list[asf_search.ASFSearchResults]) -> pystac.ItemCollection:
        grouped = defaultdict(list)
        for granule in granules:
            group_id = getattr(granule, 'properties', {}).get('groupID')
            if not group_id: 
                group_id = getattr(granule, 'properties', {}).get('fileID')
            grouped[group_id].append(granule)
        items = [self.create_item(granule) for granule in grouped.items()]
        item_collection = pystac.item_collection.ItemCollection(items, clone_items=False)
        
        return item_collection
    
    def create_item(self, grouped_granule: dict) -> pystac.Item:
        """
        Create a STAC item from a Granule object.
        Args:
            granule (dict): The grouped granules of assets to convert into a STAC item.
        Returns:
            pystac.Item: The created STAC item.
        """
        groupID, content = grouped_granule
        item_id = groupID
        g_dict = content[0].__dict__
        
        item_start_datetime = get_nested(g_dict, ['umm', 'TemporalExtent', 'RangeDateTime', 'BeginningDateTime'])
        item_end_datetime = get_nested(g_dict, ['umm', 'TemporalExtent', 'RangeDateTime', 'EndingDateTime'])
        if item_start_datetime:
            item_start_datetime = datetime.fromisoformat(item_start_datetime.replace("Z", "+00:00"))
        if item_end_datetime:
            item_end_datetime = datetime.fromisoformat(item_end_datetime.replace("Z", "+00:00"))
        item_datetime = None
        if item_datetime:
            item_datetime = datetime.fromisoformat(item_datetime.replace("Z", "+00:00"))

        if not item_start_datetime or not item_end_datetime:
            print(item_start_datetime, item_end_datetime)
            print('item differs in datetime, implement for: ', get_nested(g_dict, ['umm', 'TemporalExtent']))

        item_geometry = g_dict['geometry']
        item_bbox = list(shapely.from_geojson(str(item_geometry).replace("'",'"')).bounds)

        assets = {
            entry.properties['fileID'].split('_')[-1]:
            pystac.Asset(
                href = entry.properties['url'],
                #roles = ["data"]if entry.properties['url'].endswith(('.h5', '.tiff', '.tif', '.nc')) else ["meta"],
                ) for entry in content
                }

        del g_dict['session']

        item = pystac.Item(
            id=item_id,
            datetime=item_datetime,
            start_datetime=item_start_datetime,
            end_datetime=item_end_datetime,
            geometry=item_geometry,
            bbox=item_bbox,
            properties={'original_result': g_dict},  # needed for download? -> could be just href
            assets=assets
        )
        
        #item.validate()

        return item

    @staticmethod
    def download_urls(urls, out_dir, threads, **kwargs):
        asf_search.download.download_urls(urls, path=out_dir, processes=threads, **kwargs)

    def load_granule(self, item: pystac.Item, variables:list[str]|None=None,
                     out_dir:str | None = None, threads:int = 1, **kwargs) -> bytes:
        if variables:
            urls = [asset.href for name, asset in item.get_assets().items() if name in variables]
        else:    
            urls = [asset.href for asset in item.get_assets().values()]
        if not out_dir:
            out_dir = ''

        self.download_urls(urls, out_dir, threads, **kwargs)
        local_files = [out_dir + Path(url).name for url in urls]
        for i, lf in enumerate(local_files):
            p = Path(lf)
            if not p.is_absolute():
                local_files[i] = str(p.resolve())
        
        item = update_stac_item(item, urls, local_files)
        return item

    def create_cube(self, parameters):
        data = odc.stac.load(**parameters)
        if data is None:
            raise ValueError("Failed to create cube")
        return data


register_provider('asf', ASFProvider)

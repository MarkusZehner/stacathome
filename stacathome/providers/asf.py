from datetime import datetime

import asf_search
import odc
import odc.stac
import pystac
import shapely

from .common import BaseProvider, register_provider


class ASFProvider(BaseProvider):
    
    def __init__(self):
        super().__init__('asf')

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

        # TODO: Convert to Stac

        return results

    @staticmethod
    def download_from_asf(urls, path, **kwargs):
        asf_search.download.download_urls(urls, path=path, **kwargs)

    def create_cube(self, parameters):
        data = odc.stac.load(**parameters)
        if data is None:
            raise ValueError("Failed to create cube")
        return data


register_provider(ASFProvider)

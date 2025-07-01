from datetime import datetime

import earthaccess
import odc
import odc.stac
import pystac
import shapely

from .common import BaseProvider, register_provider


class EarthAccessProvider(BaseProvider):
    def __init__(self):
        earthaccess.login(persist=True)

    def _request_items(
        self,
        collection: str,
        starttime: datetime,
        endtime: datetime,
        area_of_interest: shapely.Geometry = None,
        limit: int = None,
        **kwargs,
    ) -> pystac.ItemCollection:
        granules = earthaccess.search_data(
            short_name=collection,
            temporal=(starttime, endtime),
            bounding_box=area_of_interest.bounds if area_of_interest else None,
            count=limit if limit else -1,
            **kwargs,
        )
        return granules

    def download_from_earthaccess(cls, granules, local_path, threads, **kwargs):
        earthaccess.download(granules, local_path=local_path, threads=threads, **kwargs)

    def create_cube(self, parameters):
        data = odc.stac.load(**parameters)
        if data is None:
            raise ValueError("Failed to create cube")
        return data


register_provider('earthaccess', EarthAccessProvider)

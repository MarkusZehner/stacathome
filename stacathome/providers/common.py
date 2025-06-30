from datetime import datetime

from odc.stac import load


class BaseProvider:
    

    def request_items(
        self, 
        collection: str, 
        starttime: datetime, 
        location: any = None, 
        max_retry: int = 5, 
        **kwargs
    ):
        raise NotImplementedError

    def download_granules_to_file(self, href_path_tuples: list[tuple]):#
        raise NotImplementedError

    def download_cube(self, parameters):
        raise NotImplementedError
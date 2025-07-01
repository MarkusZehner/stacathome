import earthaccess
import odc
import odc.stac

from .common import BaseProvider, register_provider


class EarthAccessProvider(BaseProvider):
    def __init__(self):
        earthaccess.login(persist=True)

    def request_items(self, request_time: str, request_place, **kwargs):
        bounds = request_place.bounds
        start_time, end_time = request_time.split('/')
        granules = earthaccess.search_data(temporal=(start_time, end_time), bounding_box=bounds, **kwargs)
        return granules

    def download_from_earthaccess(cls, granules, local_path, threads, **kwargs):
        earthaccess.download(granules, local_path=local_path, threads=threads, **kwargs)

    def create_cube(self, parameters):
        data = odc.stac.load(**parameters)
        if data is None:
            raise ValueError("Failed to create cube")
        return data


register_provider('earthaccess', EarthAccessProvider)

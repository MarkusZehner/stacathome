import asf_search
import odc
import odc.stac

from .common import BaseProvider

class ASFProvider:

    def request_items(request_time, request_place, **kwargs):
        wkt = request_place.wkt
        if '/' in request_time:
            start_time, end_time = request_time.split('/')
        else:
            raise ValueError('ASF (probably) requires start and end time in form of yyyy-mm-dd/yyyy-mm-dd')

        results = asf_search.search(
            start=start_time,
            end=end_time,
            intersectsWith=wkt,
            **kwargs,
        )
        return results

    def download_from_asf(urls, path, **kwargs):
        download_urls(urls, path=path, **kwargs)

    def create_cube(self, parameters):
        data = odc.stac.load(**parameters)
        if data is None:
            raise ValueError("Failed to create cube")
        return data



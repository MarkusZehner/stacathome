from pathlib import Path
from stacathome.redo_classes.requests import STACRequest
from stacathome.redo_classes.generic_utils import parse_dec_to_lon_lat_point, metric_buffer, get_utm_crs_from_lon_lat
from stacathome.redo_classes.registry import get_processor

if __name__ == '__main__':

    point = parse_dec_to_lon_lat_point('30.06579674058176, -110.308981129602946')
    p_boxed = metric_buffer(point, 1000, return_box=True)

    bucket_crs = get_utm_crs_from_lon_lat(point.x, point.y)

    collections = ['sentinel-2-l2a', 'modis-13Q1-061', 'esa-worldcover', 'sentinel-1-rtc', 'sentinel-3-synergy-syn-l2-netcdf']
    subset_bands = {
        'sentinel-2-l2a': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07'],
        'sentinel-1-rtc': ['vv', 'vh'],
        'modis-13Q1-061': ['250m_16_days_EVI'],
        'esa-worldcover': ['map'],
        "sentinel-3-synergy-syn-l2-netcdf": ['syn-s3n-reflectance']
    }
    time_ranges = {
        'sentinel-2-l2a': '2021-01-01/2021-02-15',
        'sentinel-1-rtc': '2021-01-01/2021-02-15',
        'sentinel-3-synergy-syn-l2-netcdf': '2021-01-01/2021-02-15',
        'modis-13Q1-061': '2016-01-01/2016-12-31',
        'esa-worldcover': '2021'
    }
    request = STACRequest(collections, p_boxed, time_ranges)
    items = request.request_items_basic()

    items['sentinel-2-l2a'] = [i for i in items['sentinel-2-l2a'] if get_processor(i).get_crs() == bucket_crs]

    geoboxes = request.create_geoboxes(items)
    cubes = request.load_cubes_basic(items, geoboxes,
                                     path=Path('/Net/Groups/BGI/work_5/scratch/EU_Minicubes/_test2'))

    # # alternative with checking for tile names, does not work with sentinel-3 and -1
    # request = STACRequest(collections, p_boxed, time_ranges)
    # returned_items_per_collection = request.collect_covering_tiles_and_coverage()
    # requested_items = request.request_items(returned_items_per_collection)
    # cubes_split_loaded = request.load_cubes(requested_items, returned_items_per_collection,
    #                                         subset_bands_per_collection=subset_bands,
    #                                         split_by=150,
    #                                         path=Path('/Net/Groups/BGI/work_5/scratch/EU_Minicubes/_test2'))

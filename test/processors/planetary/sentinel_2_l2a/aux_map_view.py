import folium
import shapely

from stacathome.processors.sentinel2 import (
    s2_pc_filter_coverage,
    s2_pc_filter_newest_processing_time,
    S2Item,
)
from stacathome.providers import get_provider
from .test_processor import create_test_geobox


if __name__ == '__main__':
    m = folium.Map()

    provider = get_provider('planetary_computer')

    geobox = create_test_geobox(shapely.Point(710800, 5901040), resolution=100, size_box=10000, crs='EPSG:32632')
    area_of_interest = geobox.footprint('EPSG:4326', buffer=0, npoints=4)
    m = area_of_interest.explore(m, simplify=0, name='roi', tooltip='roi',
        style_function=lambda feature: {
            'fillColor': 'red',
            'color': 'red',
            'weight': 2,
            'fillOpacity': 0.1
        })

    geobox_small = create_test_geobox(
        shapely.Point(740800, 5901040), resolution=100, size_box=100, crs='EPSG:32632'
    )
    roi_small = geobox_small.footprint('EPSG:4326', buffer=0, npoints=4)
    m = roi_small.explore(m, simplify=0, name='roi_small', tooltip='roi_small',
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 2,
            'fillOpacity': 0.1
        })

    items = provider.request_items(
        collection='sentinel-2-l2a',
        starttime='2023-07-10',
        endtime='2023-07-30',
        roi=area_of_interest,
    )
    s2_items = [S2Item(item) for item in items]


    only_newer_processing = s2_pc_filter_newest_processing_time(s2_items)

    for i in range(4):
        m = only_newer_processing[i].geometry_odc_geometry.explore(m, simplify=0, name=f'new_{i}', tooltip=f'new_{i}',
            style_function=lambda feature: {
                'fillColor': 'blue',
                'color': 'blue',
                'weight': 2,
                'fillOpacity': 0.1
            })

    coverage_filtered_items = s2_pc_filter_coverage(only_newer_processing, roi_small)

    for i in range(4):
        m = coverage_filtered_items[i].geometry_odc_geometry.explore(m, simplify=0, name=f'cov_{i}', tooltip=f'cov_{i}',
            style_function=lambda feature: {
                'fillColor': 'black',
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.1
            })

    print(set([g.id for g in s2_items]))
    print(set([g.id for g in only_newer_processing]))
    print(set([g.id for g in coverage_filtered_items]))
    m

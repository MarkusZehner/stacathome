import pytest

import odc.geo.geom as geom
import stacathome.geo as geo


def equal_area_box(left, bottom, right, top):
    return geom.box(left, bottom, right, top, crs='EPSG:8857')  # equal earth



class TestGeo:

    def test_wgs84_geodesic_distance(self):
        point1 = geom.point(0, 0, crs='EPSG:4326')
        point2 = geom.point(1.0, 0, crs='EPSG:4326')
        point3 = geom.point(0, 1.0, crs='EPSG:4326')

        # same point
        assert geo.wgs84_geodesic_distance(point1, point1) == pytest.approx(0.0)

        # 1 deg along equator
        distance = geo.wgs84_geodesic_distance(point1, point2)
        reference = 111319.4908
        assert distance == pytest.approx(reference)

        # 1 deg along meridian
        distance = geo.wgs84_geodesic_distance(point1, point3)
        reference = 110574.3886
        assert distance == pytest.approx(reference)


    def test_wgs84_geodesic_area(self):
        geometry = equal_area_box(0, 0, 10_000, 10_000)
        assert geo.wgs84_geodesic_area(geometry) == pytest.approx(10_000**2)


    def test_intersection_over_union(self):
        geom1 = equal_area_box(0, 0, 20_000, 20_000)
        geom2 = equal_area_box(1000, 1000, 21_000, 21_000)

        expected = 19_000**2 / (2*20_000**2 - 19_000**2)
        assert geo.wgs84_intersection_over_union(geom1, geom2) == pytest.approx(expected)

    
    def test_overlap_percentage(self):
        geom1 = equal_area_box(0, 0, 20000, 20000)
        geom2 = equal_area_box(5000, 5000, 25000, 25000)

        expected = (15_000**2) / 20_000**2
        assert geo.wgs84_overlap_percentage(geom1, geom2) == pytest.approx(expected)


    def test_overlap_percentage_disjunct(self):
        geom1 = equal_area_box(0, 0, 20000, 20000)
        geom2 = equal_area_box(20001, 20001, 20002, 20002)
        assert geo.wgs84_overlap_percentage(geom1, geom2) == 0.0

    
    def test_overlap_percentage_very_small(self):
        geom1 = equal_area_box(0, 0, 20000, 20000)
        geom2 = equal_area_box(19998, 19998, 20002, 20002)

        expected = (2**2) / 4**2
        assert geo.wgs84_overlap_percentage(geom1, geom2) == pytest.approx(expected)

    
    def test_contains(self):
        geom1 = equal_area_box(0, 0, 20000, 20000)
        geom2 = equal_area_box(1000, 1000, 2000, 2000)
        assert geo.wgs84_contains(geom1, geom2)
        assert not geo.wgs84_contains(geom2, geom1)


    def test_contains_partial_overlap(self):
        geom1 = equal_area_box(0, 0, 1000, 1000)
        geom2 = equal_area_box(500, 500, 2000, 2000)
        assert not geo.wgs84_contains(geom1, geom2)
        assert not geo.wgs84_contains(geom2, geom1)

    
    def test_intersects(self):
        geom1 = equal_area_box(0, 0, 1000, 1000)
        
        p1 = geom.point(999.5, 999.5, crs='EPSG:8857').to_crs('EPSG:4326')
        p2 = p1.transform(lambda x,y: (x+2, y+2))
        p3 = p1.transform(lambda x,y: (x+2, y))
        polygon = geom.polygon([p1.points[0], p2.points[0], p3.points[0], p1.points[0]], crs='EPSG:4326')
        assert geo.wgs84_intersects(geom1, polygon)
        assert geo.wgs84_intersects(polygon, geom1)

    
    def test_intersects_disjunct(self):
        geom1 = equal_area_box(0, 0, 1000, 1000)
        p1 = geom.point(1000.1, 1000.1, crs='EPSG:8857').to_crs('EPSG:4326')
        assert not geo.wgs84_intersects(geom1, p1)
        assert not geo.wgs84_intersects(p1, geom1)
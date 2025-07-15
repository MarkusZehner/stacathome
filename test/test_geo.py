import shapely

import odc.geo.geom as geom
from stacathome.geo import wgs84_centroid, wgs84_geodesic_distance


class TestGeo:

    def test_wgs84_geodesic_distance(self):
        """
        Test the wgs84_geodesic_distance function.
        """
        point1 = geom.point(0, 0, crs='EPSG:4326')
        point2 = geom.point(1.0, 0, crs='EPSG:4326')
        point3 = geom.point(0, 1.0, crs='EPSG:4326')

        # same point
        assert wgs84_geodesic_distance(point1, point1) < 1e-9

        # 1 deg along equator
        distance = wgs84_geodesic_distance(point1, point2)
        reference = 111319.4908
        assert abs(distance - reference) < 1e-2

        # 1 deg along meridian
        distance = wgs84_geodesic_distance(point1, point3)
        reference = 110574.3886
        assert abs(distance - reference) < 1e-2

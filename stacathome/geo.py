from functools import partial

import numpy as np
import odc.geo.geom as geom
import pyproj
import pystac
import shapely


_WGS84_GEOD = pyproj.Geod(ellps='WGS84')


def to_equal_area(geometry: geom.Geometry, resolution=None) -> geom.Geometry:
    """
    Transform a geometry to an equal area projection. In particular, a Sinusoidal projection (ESRI:54008) is used.

    Args:
        geometry (Geometry): The geometry to transform.

    Returns:
        Geometry: The transformed geometry in Sinusoidal projection (ESRI:54008).
    """
    return geometry.to_crs('ESRI:54008', resolution=resolution)


def to_wgs84(geometry: geom.Geometry, resolution=None) -> geom.Geometry:
    """
    Transform a geometry to WGS 84 (EPSG:4326).

    Args:
        geometry (Geometry): The geometry to transform.

    Returns:
        Geometry: The transformed geometry in WGS 84 (EPSG:4326).
    """
    return geometry.to_crs('EPSG:4326', resolution=resolution)


def wgs84_centroid(geometry: geom.Geometry) -> geom.Geometry:
    """
    Calculate the centroid of a geometry in WGS 84 (EPSG:4326) coordinate reference system.
    This function takes into account the geometry of the spheroid and returns the centroid in latitude and longitude.

    Args:
        geometry (Geometry): The geometry for which to calculate the centroid.
    Returns:
        shapely.Point: The centroid of the geometry in WGS 84 (EPSG:4326) coordinate reference system.
    """
    # TODO: Implement "A New Method for Finding Geographic Centers, with Application to U.S. States", Peter A. Rogerson
    return geometry.centroid.to_crs('EPSG:4326')


def wgs84_geodesic_distance(point1: geom.Geometry, point2: geom.Geometry) -> float:
    """
    Calculate the distance between two points in WGS 84 (EPSG:4326) coordinate reference system.

    Args:
        point1 (Geometry): The first point.
        point2 (Geometry): The second point.
    Returns:
        float: The distance between the two points in meters.
    """
    if point1.geom_type != 'Point' or point2.geom_type != 'Point':
        raise ValueError(f"Only point geometries are accepted. Got: '{point1.geom_typ}' and '{point2.geom_typ}'")
    point1 = to_wgs84(point1)
    point2 = to_wgs84(point2)
    xx = (shapely.get_x(point1.geom), shapely.get_x(point2.geom))
    yy = (shapely.get_y(point1.geom), shapely.get_y(point2.geom))
    return _WGS84_GEOD.line_length(xx, yy)


def wgs84_geodesic_area(geometry: geom.Geometry) -> float:
    """
    Computes the geodesic area of a polygon on the WGS 84 reference ellipsoid.

    Args:
        geometry (Geometry): The input geometry object representing the polygon.

    Returns:
        float: The geodesic area of the polygon in square meters.
    """
    geometry = to_wgs84(geometry)
    coords = shapely.get_coordinates(geometry.geom)
    area, _ = _WGS84_GEOD.polygon_area_perimeter(coords[:, 1], coords[:, 0])
    return area


def wgs84_centroid_distance(shape1: geom.Geometry, shape2: geom.Geometry) -> float:
    """
    Calculate the geodesic distance between the centroids of two shapes with regard to the WGS 84 reference ellipsoid.

    Args:
        shape1 (Geometry): The first shape.
        shape2 (Geometry): The second shape.
    
    Returns:
        float: The distance between the centroids of the two shapes in meters.
    """
    centroid1 = wgs84_centroid(shape1)
    centroid2 = wgs84_centroid(shape2)
    return wgs84_geodesic_distance(centroid1, centroid2)


def wgs84_intersection_over_union(shape1: geom.Geometry, shape2: geom.Geometry) -> float:
    """
    Calculate the geodesic Intersection over Union (IoU) percentage between two geometries.

    Args:
        shape1 (Geometry): The first shape.
        shape2 (Geometry): The second shape.

    Returns:
        float: The IoU percentage between the two shapes. Returns 0.0 if there is no intersection.
    """
    shape1 = to_equal_area(shape1)
    shape2 = to_equal_area(shape2)
    union = shape1.union(shape2)
    intersection = shape1.intersection(shape2)
    return intersection.area / union.area if intersection.area > 0 else 0.0


def wgs84_overlap_percentage(shape1: geom.Geometry, shape2: geom.Geometry) -> float:
    """
    Calculate the geodesic overlap percentage (measured from the smaller shape) between two shapes.

    Args:
        shape1 (Geometry): The first shape.
        shape2 (Geometry): The second shape.

    Returns:
        float: The overlap percentage between the two shapes. 0.0 if there is no intersection, 1.0 if one shape is fully contained in the other.
    """

    shape1 = to_equal_area(shape1)
    shape2 = to_equal_area(shape2)
    intersection = shape1.intersection(shape2)
    if intersection.is_empty:
        return 0.0
    area = min(shape1.area, shape2.area)
    return intersection.area / area if area > 0 else 0.0


def wgs84_contains(geometry1: geom.Geometry, geometry2: geom.Geometry) -> bool:
    """
    Check if one geometry fuly contains another geometry when interpreted as geodesic polygons on the WGS 84 reference ellipsoid. 

    Args:
        shape1 (Geometry): The first geometry.
        shape2 (Geometry): The second geometry.
    Returns:
        bool: True if geometry1 fully contains geometry2, False otherwise.
    """
    geometry1 = to_equal_area(geometry1)
    geometry2 = to_equal_area(geometry2)
    return geometry1.contains(geometry2)

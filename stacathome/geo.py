from functools import partial

import numpy as np
import odc.geo.geom as geom
import pyproj
import pystac
import shapely

_WGS84_GEOD = pyproj.Geod(ellps='WGS84')


def to_equal_area(geometry: geom.Geometry) -> geom.Geometry:
    """
    Transform a geometry to Lambert Cylindrical Equal Area (EPSG:9835).
    This is useful for area calculations in a projected coordinate system.

    Args:
        geometry (Geometry): The geometry to transform.
    Returns:
        Geometry: The transformed geometry in Lambert Cylindrical Equal Area (EPSG:9835).
    """
    return geometry.to_crs('EPSG:9835')


def to_wgs84(geometry: geom.Geometry) -> geom.Geometry:
    """
    Transform a geometry to WGS 84 (EPSG:4326).

    Args:
        geometry (Geometry): The geometry to transform.
    Returns:
        Geometry: The transformed geometry in WGS 84 (EPSG:4326).
    """
    return geometry.to_crs('EPSG:4326')


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


def centroid_distance(shape1: geom.Geometry, shape2: geom.Geometry) -> float:
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


def wgs84_IoU(shape1: geom.Geometry, shape2: geom.Geometry) -> float:
    """
    Calculate the Intersection over Union (IoU) percentage between two shapes in WGS 84 (EPSG:4326) coordinate reference system.
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


def overlap_percentage(shape1: geom.Geometry, shape2: geom.Geometry) -> float:
    """
    Calculate the overlap percentage (measured from the smaller shape) between two shapes in WGS 84 (EPSG:4326) coordinate reference system.

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


def wgs84_contains(shape1: geom.Geometry, shape2: geom.Geometry, local_proj_code: str) -> bool:
    """
    Check if one shape fuly contains another shape in WGS 84 (EPSG:4326) coordinate reference system if projected to a local coordinate system.

    Args:
        shape1 (Geometry): The first shape.
        shape2 (Geometry): The second shape.
        local_proj_code (str): The local projection code to transform the shapes before checking containment.

    Returns:
        bool: True if shape1 fully contains shape2, False otherwise.
    """
    shape1 = shape1.to_crs(local_proj_code)
    shape2 = shape2.to_crs(local_proj_code)
    return shape1.contains(shape2)

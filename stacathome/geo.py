import numpy as np
from functools import partial
import pyproj
import pystac
import shapely
import odc.geo.geom

_WGS84_GEOD = pyproj.Geod(ellps='WGS84')
_PROJ_WGS84 = pyproj.Proj('EPSG:4326')
_PROJ_LAMBERT = pyproj.Proj('EPSG:9835')  # Lambert Cylindrical Equal Area
_EQUAL_AREA_TRANSFORMER = pyproj.Transformer.from_proj(_PROJ_WGS84, _PROJ_LAMBERT, always_xy=True)


def wgs84_to_equal_area(geometry: shapely.Geometry) -> shapely.Geometry:
    """
    Transform a geometry from WGS 84 (EPSG:4326) to Lambert Cylindrical Equal Area (EPSG:9835).
    This is useful for area calculations in a projected coordinate system.

    Args:
        geometry (shapely.Geometry): The geometry to transform.
    Returns:
        shapely.Geometry: The transformed geometry in Lambert Cylindrical Equal Area (EPSG:9835).
    """
    return shapely.transform(geometry, _EQUAL_AREA_TRANSFORMER.transform, interleaved=False)


def wgs84_centroid(geometry: odc.geo.geom.Geometry) -> shapely.Point:
    """
    Calculate the centroid of a geometry in WGS 84 (EPSG:4326) coordinate reference system.
    This function takes into account the geometry of the spheroid and returns the centroid in latitude and longitude.

    Args:
        geometry (shapely.Geometry): The geometry for which to calculate the centroid.
    Returns:
        shapely.Point: The centroid of the geometry in WGS 84 (EPSG:4326) coordinate reference system.
    """
    # TODO: Implement "A New Method for Finding Geographic Centers, with Application to U.S. States", Peter A. Rogerson
    return shapely.Point(*geometry.centroid.points)


def wgs84_geodesic_distance(point1: shapely.Point, point2: shapely.Point) -> float:
    """
    Calculate the distance between two points in WGS 84 (EPSG:4326) coordinate reference system.
    The points are assumed to be in latitude and longitude.

    Args:
        point1 (shapely.Point): The first point.
        point2 (shapely.Point): The second point.
    Returns:
        float: The distance between the two points in meters.
    """
    return _WGS84_GEOD.line_length([point1.x, point2.x], [point1.y, point2.y])


def centroid_distance(shape1: odc.geo.geom.Geometry, shape2: odc.geo.geom.Geometry) -> float:
    """
    Calculate the distance between the centroids of two shapes. The shapes are assumed to be in WGS 84 (EPSG:4326) coordinate reference system.
    Args:
        shape1 (shapely.Geometry): The first shape.
        shape2 (shapely.Geometry): The second shape.
    Returns:
        float: The distance between the centroids of the two shapes in meters.
    """
    centroid1 = wgs84_centroid(shape1)
    centroid2 = wgs84_centroid(shape2)
    return wgs84_geodesic_distance(centroid1, centroid2)


def wgs84_IoU(shape1: shapely.Geometry, shape2: shapely.Geometry) -> float:
    """
    Calculate the Intersection over Union (IoU) percentage between two shapes in WGS 84 (EPSG:4326) coordinate reference system.
    Args:
        shape1 (shapely.Geometry): The first shape.
        shape2 (shapely.Geometry): The second shape.
    Returns:
        float: The IoU percentage between the two shapes. Returns 0.0 if there is no intersection.
    """
    shape1 = wgs84_to_equal_area(shape1)
    shape2 = wgs84_to_equal_area(shape2)
    union = shape1.union(shape2)
    intersection = shape1.intersection(shape2)
    return intersection.area / union.area if intersection.area > 0 else 0.0


def wgs84_overlap_percentage(shape1: shapely.Geometry, shape2: shapely.Geometry) -> float:
    """
    Calculate the overlap percentage (measured from the smaller shape) between two shapes in WGS 84 (EPSG:4326) coordinate reference system.

    Args:
        shape1 (shapely.Geometry): The first shape.
        shape2 (shapely.Geometry): The second shape.

    Returns:
        float: The overlap percentage between the two shapes. 0.0 if there is no intersection, 1.0 if one shape is fully contained in the other.
    """

    shape1 = wgs84_to_equal_area(shape1)
    shape2 = wgs84_to_equal_area(shape2)
    intersection = shape1.intersection(shape2)
    if intersection.is_empty:
        return 0.0
    area = min(shape1.area, shape2.area)
    return intersection.area / area if area > 0 else 0.0


def wgs84_contains(shape1: odc.geo.geom.Geometry, shape2: odc.geo.geom.Geometry, local_proj_code: str) -> bool:
    """
    Check if one shape fuly contains another shape in WGS 84 (EPSG:4326) coordinate reference system if projected to a local coordinate system.

    Args:
        shape1 (shapely.Geometry): The first shape.
        shape2 (shapely.Geometry): The second shape.
        local_proj_code (str): The local projection code to transform the shapes before checking containment.

    Returns:
        bool: True if shape1 fully contains shape2, False otherwise.
    """
    shape1 = shape1.to_crs(local_proj_code)
    shape2 = shape2.to_crs(local_proj_code)
    return shape1.contains(shape2)

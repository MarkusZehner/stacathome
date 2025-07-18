import odc.geo.geom as geom
import pyproj
import shapely
from odc.geo.geobox import GeoBox


_WGS84_GEOD = pyproj.Geod(ellps='WGS84')


def is_point(geometry: geom.Geometry) -> bool:
    """
    Determines whether the provided geometry is a Point.

    Returns:
        bool: True if the geometry is a Point, False otherwise.
    """
    return geometry.geom_type == 'Point'


def get_xy(point: geom.Geometry) -> tuple[float, float]:
    """
    Extracts the x and y coordinates from a given geometry point.

    Args:
        point (geom.Geometry): A geometry object containing a point.

    Returns:
        tuple[float, float]: A tuple containing the x and y coordinates of the point, nans if not a point.
    """
    return shapely.get_x(point.geom), shapely.get_y(point.geom)


def difference_vector(point1: geom.Geometry, point2: geom.Geometry) -> tuple[float, float]:
    """
    Vector between point1 and point2 expressed in the CRS of point1.‚
    """
    point2 = point2.to_crs(point1.crs)
    x1, y1 = get_xy(point1)
    x2, y2 = get_xy(point2)
    return x1 - x2, y1 - y2


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
    if not is_point(point1) or not is_point(point2):
        raise ValueError(f"Only point geometries are accepted. Got: '{point1.geom_typ}' and '{point2.geom_typ}'")
    point1 = to_wgs84(point1)
    point2 = to_wgs84(point2)
    x1, y1 = get_xy(point1)
    x2, y2 = get_xy(point2)
    return _WGS84_GEOD.line_length((x1, x2), (y1, y2))


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
    Check if one geometry fully contains another geometry when interpreted as geodesic polygons on the WGS 84 reference ellipsoid.

    Args:
        geometry1 (Geometry): The first geometry.
        geometry2 (Geometry): The second geometry.

    Returns:
        bool: True if geometry1 fully contains geometry2, False otherwise.
    """
    geometry1 = to_equal_area(geometry1)
    geometry2 = to_equal_area(geometry2)
    return geometry1.contains(geometry2)


def wgs84_intersects(geometry1: geom.Geometry, geometry2: geom.Geometry) -> bool:
    """
    Check if one geometry intersects with another geometry when interpreted as geodesic polygons on the WGS 84 reference ellipsoid.

    Args:
        geometry1 (Geometry): The first geometry.
        geometry2 (Geometry): The second geometry.

    Returns:
        bool: True if geometry1 intersects geometry2, False otherwise.
    """
    geometry1 = to_equal_area(geometry1)
    geometry2 = to_equal_area(geometry2)
    return geometry1.intersects(geometry2)


def nearest_pixel_edge(point: geom.Geometry, geobox: GeoBox, return_crs_coords: bool = False) -> tuple[int, int]:
    if not is_point(point):
        raise ValueError('Only point geometries are supported')

    x_wld, y_wld = get_xy(point.to_crs(geobox.crs))
    x_pix, y_pix = geobox.wld2pix(x_wld, y_wld)
    x_pix, y_pix = round(x_pix), round(y_pix)
    if return_crs_coords:
        return geobox.pix2wld(x_pix, y_pix)
    else:
        return x_pix, y_pix


def closest_geobox_to_point(point: geom.Geometry, geobox: GeoBox, width: int, height: int) -> GeoBox:
    if not is_point(point):
        raise ValueError('Only point geometries are supported')

    if width % 2 != 0 or height % 2 != 0:
        raise ValueError('Only even widths and heights are currently supported')

    x_pix, y_pix = nearest_pixel_edge(point, geobox)
    pixel_box = geobox.translate_pix(x_pix, y_pix).crop((1, 1))  # (1,1) Geobox with corner at x_pix, y_pix
    output_box = pixel_box.pad(width // 2, height // 2).crop(
        (width, height)
    )  # pad to (width+1, height+1) then crop the extra pixel
    return output_box

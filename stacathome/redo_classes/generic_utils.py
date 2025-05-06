import math
import numpy as np

from shapely import box


def create_utm_grid_bbox(bbox, grid_size=60):
    """
    Snap the bounding box to a utm grid of the specified pixel size.

    Parameters:
    ----------
    bbox (BoundingBox): The bounding box to snap.
        grid_size (int): The size of the grid to snap to.

    Returns:
    -------
    shapely.geometry.box: The snapped bounding box.
    """
    xmin, ymin, xmax, ymax = bbox
    xmin_snapped = math.floor(xmin / grid_size) * grid_size
    ymin_snapped = math.floor(ymin / grid_size) * grid_size
    xmax_snapped = math.ceil(xmax / grid_size) * grid_size
    ymax_snapped = math.ceil(ymax / grid_size) * grid_size
    return box(xmin_snapped, ymin_snapped, xmax_snapped, ymax_snapped)


def arange_bounds(bounds, step):
    return np.arange(bounds[0], bounds[2], step), np.arange(bounds[1], bounds[3], step)

import numpy as np
import dask
from functools import partial
from shapely.geometry import Point
from shapely import transform

from pystac_client import Client as pystacClient

def request_items_parallel(
    lon_lat, start_date, end_date, url, collection, transform_to=None
):
    p = Point(lon_lat[0], lon_lat[1])
    if transform_to is not None:
        p = transform(p, transform_to, include_z=False)
    catalog = pystacClient.open(url)
    return catalog.search(
        intersects=p,
        datetime=f"{start_date}/{end_date}",
        collections=[collection],
    ).item_collection()


def parallel_request(
    start_date, end_date, collection, url, transform_to, aoi, crs, grid_size=None
):
    """
    Request items in parallel.
    Will request items for a regular grid of points in the aoi.

    Parameters
    ----------
    start_date: str
        The start date of the request.
    end_date: str
        The end date of the request.
    grid_size: float
        The size of the grid in crs units.
        default is 0.45, ~50 km to catch S2 tiles of 110x110 km.

    Returns
    -------
    items: list
        The requested items.
    """
    # if self.crs != 4326 and (grid_size is None or grid_size < 1.):
    #     raise UserWarning(
    #         'Default grid size not suitable for non 4326 crs, please provide grid_size')
    if grid_size is None and crs == 4326:
        grid_size = 0.25
    elif grid_size is None and crs != 4326:
        grid_size = 50000
        print("Assuming units of meters and grid size of 50 km for non 4326 crs!")

    lonmin, latmin, lonmax, latmax = aoi.bounds

    resolution = grid_size
    X, Y = np.meshgrid(
        np.arange(latmin, latmax, resolution), np.arange(lonmin, lonmax, resolution)
    )

    points = list(zip(Y.flatten(), X.flatten()))

    process = [(point) for point in points if aoi.contains(Point(*point))]

    if len(process) == 0:
        process = [
            (lonmin, latmin),
            (lonmax, latmax),
            (lonmin, latmax),
            (lonmax, latmin),
        ]

    request_partial = partial(
        request_items_parallel,
        start_date=start_date,
        end_date=end_date,
        collection=collection,
        url=url,
        transform_to=transform_to,
    )
    do = []
    for p in process:
        do.append(dask.delayed(request_partial)(p))
    items = dask.compute(*do)
    # dask_bag = db.from_sequence(process).map(request_partial)
    # items = dask_bag.compute()
    return items

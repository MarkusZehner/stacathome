import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.affinity import translate
import geopandas as gpd
from tqdm import tqdm
from shapely import unary_union
from urllib.request import urlretrieve, urlopen
import io


def define_grid_along_axis(len_bounds: int, grid_size: int) -> tuple[int, int, int]:
    """
    Calculate the start, end, and offset of one UTM tile along one axis.
    The offset is used to center the grid within the bounds.

    Parameters
    ----------
    len_bounds : int
        Length of the bounds along the axis.
    grid_size : int
        Size of the grid.

    Returns
    -------
    tuple[int, int, int]
        Start, end, and offset of the grid.
    """
    n_units = len_bounds // grid_size
    encompassing_bounds_in_grid = (n_units + 1) * grid_size
    difference = len_bounds - encompassing_bounds_in_grid
    if np.round(encompassing_bounds_in_grid / len_bounds, 0) == 1.0:
        offset = difference
        start = 1
    else:
        offset = grid_size - np.abs(difference)
        start = 0
    offset = offset // 2
    return start, n_units, offset


def fishnet_s2_utm_tile(
    tile: gpd.GeoDataFrame,
    xsize: int,
    ysize: int,
    min_overlap: int = 5000,
    create_template: bool = False,
) -> gpd.GeoDataFrame:
    """
    Function to create a fishnet of a Sentinel-2 UTM tile with given x and y size.
    min_overlap assures that there are no gaps between the tiles.
    create_template creates a fishnet with the same size as the tile,
    but with the origin at 0,0, to be shifted around for faster computation

    Parameters
    ----------
    tile : gpd.GeoDataFrame
        Sentinel-2 UTM tile.
    xsize : int
        Size of the grid along the x-axis.
    ysize : int
        Size of the grid along the y-axis.
    min_overlap : int, optional
        Minimum overlap between the tiles, by default 5000.
    create_template : bool, optional
        Create a template fishnet, by default False.

    Returns
    -------
    gpd.GeoDataFrame
        Fishnet of the Sentinel-2 UTM tile.
    """
    bounds = [
        float(b)
        for b in tile["utm_bounds"].replace("(", "").replace(")", "").split(",")
    ]
    x_size_actual = bounds[2] - bounds[0]
    y_size_actual = bounds[3] - bounds[1]

    x_start, x_end, x_offset = define_grid_along_axis(x_size_actual, xsize)
    if x_size_actual == y_size_actual and xsize == ysize:
        y_start, y_end, y_offset = x_start, x_end, x_offset
    else:
        y_start, y_end, y_offset = define_grid_along_axis(y_size_actual, ysize)

    if abs(x_offset) > min_overlap or abs(y_offset) > min_overlap:
        raise ValueError(
            f"Overlap is too little, leads to missing coverage: {x_offset}, {y_offset}"
        )
    if abs(x_offset) % 60 != 0 or abs(y_offset) % 60 != 0:
        raise ValueError(
            f"Overlap not divisible by 60, requires resampling of coarse bands: {x_offset}, {y_offset}"
        )

    if create_template:
        bounds[0] = 0
        bounds[1] = 0

    # create the fishnet
    fishnet = []
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            x = bounds[0] + x_offset + i * xsize
            y = bounds[1] + y_offset + j * ysize
            # create the polygon
            poly = Polygon(
                [(x, y), (x + xsize, y), (x + xsize, y + ysize), (x, y + ysize), (x, y)]
            )
            fishnet.append(poly)
    return gpd.GeoDataFrame(
        {"tile": [f"{tile['tile']}_{n}" for n in range(len(fishnet))]},
        geometry=fishnet,
        crs=tile["epsg"],
    )


def apply_template(fishnet_template: gpd.GeoDataFrame, tile: str):
    """
    Wrapper to apply the fishnet template to a UTM tile.
    """
    fishnet_template = fishnet_template.copy()
    bounds = [
        float(b)
        for b in tile["utm_bounds"].replace("(", "").replace(")", "").split(",")
    ]
    shifted_geometries = fishnet_template["geometry"].apply(
        lambda geom: translate(geom, xoff=bounds[0], yoff=bounds[1])
    )
    fishnet_template["geometry"] = shifted_geometries
    fishnet_template["tile"] = fishnet_template["tile"].apply(
        lambda tile_name: tile["tile"] + tile_name[5:]
    )
    return fishnet_template


def apply_fishnet_to_utm_grid_zone(
    utm_grid: gpd.GeoDataFrame, zone: int, gridsize: int
) -> gpd.GeoDataFrame:
    """
    Wrapper to create a fishnet for all tiles of one UTM zone.

    Parameters
    ----------
    utm_grid : gpd.GeoDataFrame
        Sentinel-2 UTM grid.
    zone : int
        UTM zone.
    gridsize : int
        Size of the grid.

    Returns
    -------
    gpd.GeoDataFrame
        Fishnet of the Sentinel-2 UTM zone.
    """
    utm_grid_part = utm_grid.where(utm_grid["epsg"] == zone).dropna().copy()
    utm_grid_part.set_crs(epsg=zone, inplace=True, allow_override=True)
    template = fishnet_s2_utm_tile(
        utm_grid_part.iloc[0], gridsize, gridsize, create_template=True
    )
    fish = utm_grid_part.apply(lambda tile: apply_template(template, tile), axis=1)
    fish = gpd.GeoDataFrame(pd.concat(fish.to_list(), ignore_index=True), crs=zone)
    fish["epsg"] = fish.crs.to_epsg()
    fish["utm_wkt"] = fish["geometry"].apply(lambda geom: geom.wkt)
    fish["utm_bounds"] = fish["geometry"].apply(lambda geom: geom.bounds)
    fish.to_crs(4326, inplace=True)

    return fish


def get_natural_earth_landcover(
    filename: str = "natural_earth_vector_ne_10m_land.parquet",
) -> gpd.GeoDataFrame:
    """
    Download the Natural Earth 10m landcover dataset and return it as a geopandas dataframe,
    or load it from disk if already downloaded and processed.

    Parameters
    ----------
    temp_folder : str, optional

    filename : str, optional

    """
    if not os.path.exists(filename):
        geojson_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson"
        response = urlopen(geojson_url)
        geojson_data = io.BytesIO(response.read())
        gdf = gpd.read_file(geojson_data)
        gdf.to_parquet(filename)

    return gpd.read_parquet(filename)


def get_sentinel2_grid(
    filenames: tuple[str] = ("sentinel-2-grid.parquet", "sentinel-2-grid_LAND.parquet"),
) -> gpd.GeoDataFrame:
    """
    download the Sentinel-2 UTM grid dataset and return it as a geopandas dataframe
    or load from disk if already downloaded

    Parameters
    ----------
    filenames : tuple[str], optional
        Filenames of the Sentinel-2 UTM grid datasets, by default ("sentinel-2-grid.parquet", "sentinel-2-grid_LAND.parquet").

    Returns
    -------
    gpd.GeoDataFrame
        Sentinel-2 UTM grid datasets.
    """
    if not os.path.exists(filenames[0]):
        urlretrieve(
            "https://github.com/maawoo/sentinel-2-grid-geoparquet/blob/main/sentinel-2-grid.parquet?raw=true",
            filenames[0],
        )
    if not os.path.exists(filenames[1]):
        urlretrieve(
            "https://github.com/maawoo/sentinel-2-grid-geoparquet/blob/main/sentinel-2-grid_LAND.parquet?raw=true",
            filenames[1],
        )
    return gpd.read_parquet(filenames[0]), gpd.read_parquet(filenames[1])


def create_global_buckets(
    gridsize_m: int = 5040,
    temp_folder: str = "bucket_temp",
    aux_folder: str = None,
    out_file_name: str = "S2_buckets_world",
    sentinel2_grid: str = "sentinel-2-grid.parquet",
    sentinel2_grid_land: str = "sentinel-2-grid_LAND.parquet",
    natural_earth_land: str = "natural_earth_vector_ne_10m_land.parquet",
) -> None:
    """
    Function to create a global fishnet of Sentinel-2 UTM tiles, where the fishnet is clipped to the land area.
    The current implementation just clips the UTM zones from the right.
    Intermediate results for each UTM zone are stored in the temp_folder.

    Parameters
    ----------
    gridsize_m : int, optional
        Size of the grid in meters, by default 5040.
    temp_folder : str, optional
        Temporary folder to store the fishnet parts, by default "bucket_temp".
    out_file : str, optional
        Output file for the global fishnet, by default "S2_buckets_world.parquet".
    sentinel2_grid : str, optional
        Path to the Sentinel-2 UTM grid, by default "sentinel-2-grid.parquet".
    sentinel2_land : str, optional
        Path to the Sentinel-2 UTM land grid, by default "sentinel-2-grid_LAND.parquet".
    natural_earth_land : str, optional
        Path to the Natural Earth land grid, by default "NaturalEarthLand10m.parquet".
    Returns
    -------
    None

    """
    if aux_folder is not None:
        os.makedirs(temp_folder, exist_ok=True)
        sentinel2_grid = os.path.join(aux_folder, sentinel2_grid)
        sentinel2_grid_land = os.path.join(aux_folder, sentinel2_grid_land)
        natural_earth_land = os.path.join(aux_folder, natural_earth_land)

    sentinel2_grid = get_sentinel2_grid(sentinel2_grid)
    sentinel2_grid_land = get_sentinel2_grid(sentinel2_grid_land)
    natural_earth_land = unary_union(
        get_natural_earth_landcover(natural_earth_land)["geometry"]
    )

    out_file = f"{out_file_name}_grid_{gridsize_m}.parquet"
    out_file = os.path.join(aux_folder, out_file)
    if os.path.exists(out_file):
        return

    os.makedirs(temp_folder, exist_ok=True)
    sentinel2_grid_land_enlarged = sentinel2_grid.where(
        sentinel2_grid.intersects(unary_union(sentinel2_grid_land["geometry"]))
    ).dropna()

    epsgs = sentinel2_grid["epsg"].unique()
    epsgs.sort()

    epsgs_n = epsgs[epsgs < 32700]
    epsgs_s = epsgs[epsgs > 32700]

    for half in ["N", "S"]:
        if half == "N":
            epsgs_list = epsgs_n
        else:
            epsgs_list = epsgs_s

        for i in tqdm(range(len(epsgs_list))):
            # we take a by one tile enlarged version to clip to remove the overlap at coastlines
            previous_n = apply_fishnet_to_utm_grid_zone(
                sentinel2_grid_land_enlarged, epsgs_n[i - 1], gridsize_m * 20
            )  # 20 times larger equals one large bucket with the same size as the fishnet
            previous_s = apply_fishnet_to_utm_grid_zone(
                sentinel2_grid_land_enlarged, epsgs_s[i - 1], gridsize_m * 20
            )
            previous_n = unary_union(
                previous_n.to_crs(epsgs_n[i - 1]).buffer(0.01).to_crs(4326).buffer(0)
            )
            previous_s = unary_union(
                previous_s.to_crs(epsgs_s[i - 1]).buffer(0.01).to_crs(4326).buffer(0)
            )

            # combine the two previous N and S utm zones
            previous = unary_union([previous_n, previous_s])

            # fishnet the land tiles, clip to the previous larger, then the towards the actual land area
            fish_temp = apply_fishnet_to_utm_grid_zone(
                sentinel2_grid_land, epsgs_list[i], gridsize_m
            )
            fish_temp = fish_temp.where(
                fish_temp.apply(
                    lambda row: not Polygon(row["geometry"]).within(previous), axis=1
                )
            ).dropna()
            fish_temp = fish_temp.where(
                fish_temp.apply(
                    lambda row: Polygon(row["geometry"]).intersects(natural_earth_land),
                    axis=1,
                )
            ).dropna()
            fish_temp.to_parquet(
                os.path.join(temp_folder, f"fishnet_part_{epsgs_list[i]}.parquet"),
                engine="pyarrow",
            )

    fish_inland = gpd.GeoDataFrame(
        pd.concat(
            [
                gpd.read_parquet(
                    os.path.join(temp_folder, f"fishnet_part_{epsg}.parquet")
                )
                for epsg in epsgs
                if os.path.exists(
                    os.path.join(temp_folder, f"fishnet_part_{epsg}.parquet")
                )
            ],
            ignore_index=True,
        ),
        crs=4326,
    )
    fish_inland.to_parquet(out_file, engine="pyarrow")


if __name__ == "__main__":
    create_global_buckets(aux_folder=None)

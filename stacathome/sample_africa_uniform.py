import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union


def sample_points(gdf, n_points, seed):
    rng = np.random.default_rng(seed)

    geometry = unary_union(gdf.geometry)
    min_x, min_y, max_x, max_y = geometry.bounds

    samples = gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    while len(samples) < n_points:
        x_candidates = rng.uniform(min_x, max_x, size=int(n_points * 1.5))
        y_candidates = rng.uniform(min_y, max_y, size=int(n_points * 1.5))

        points = [Point(x, y) for x, y in zip(x_candidates, y_candidates)]
        points = gpd.GeoDataFrame(geometry=points, crs=gdf.crs)
        points = points[points.intersects(geometry)]  # only keep points within the polygon

        samples = gpd.GeoDataFrame(pd.concat([samples, points], ignore_index=True), crs=gdf.crs)

    return samples[:n_points]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vectorfile',
        type=str,
        default='/Net/Groups/BGI/scratch/mzehner/code/stacathome/aux_data/FAO_Africa_adm0_Country.parquet',
    )
    parser.add_argument('--outfile', type=str, default='static/proto_dataset/random_points.csv')
    parser.add_argument('--npoints', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    gdf = gpd.read_parquet("/Net/Groups/BGI/scratch/mzehner/code/stacathome/aux_data/FAO_Africa_adm0_Country.parquet")
    gdf_dissolved = gdf.dissolve()  # dissolve all geometries into one (union)
    gdf_projected = gdf_dissolved.to_crs(epsg=27701)  # reproject to Equi7 Africa

    gdf_samples = sample_points(gdf_projected, args.npoints, seed=args.seed)
    gdf_samples = gdf_samples.to_crs(epsg=4326)  # back to latlon

    df = pd.DataFrame(np.stack([gdf_samples.geometry.y, gdf_samples.geometry.x], axis=1), columns=['lat', 'lon'])
    df.to_csv(args.outfile, index=False)


if __name__ == '__main__':
    main()

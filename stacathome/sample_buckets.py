import argparse
import random

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point


from timegrid import WT15
from spatialgrid import UTM5KM


def load_points(paths):
    point_dfs = []
    for point_file in paths:
        point_dfs += [pd.read_csv(point_file)]
    point_dfs = pd.concat(point_dfs, ignore_index=True)
    points = [Point(lon, lat) for lon, lat in zip(point_dfs.lon, point_dfs.lat)]
    return point_dfs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucketfile', type=str, default='')
    parser.add_argument('--outfile', type=str, default='static/proto_dataset/')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--n', type=int, default=60000)
    parser.add_argument('--startdate', type=str, default='2016-01-01')
    parser.add_argument('--enddate', type=str, default='2021-01-01')
    parser.add_argument('points', nargs='+', type=str)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    points_df = load_points(args.points)

    grid_cells = set()
    for lat,lon in zip(points_df.lat, points_df.lon):
        cells = UTM5KM.find_latlon(lat,lon)
        if len(cells) > 1:
            cell = rng.choice(cells)
            grid_cells.add(cell)
        elif cells:
            grid_cells.add(cells[0])


    print(len(grid_cells))

    left_idx = WT15.date_to_index(args.startdate)
    right_idx = WT15.date_to_index(args.enddate)
    time_indices = rng.integers(left_idx, right_idx, size=len(grid_cells) * 2)
    print(time_indices)


    #bucketlist_africa.to_parquet("/Net/Groups/BGI/scratch/mzehner/code/stacathome/aux_data/proto_bucketlist_africa.parquet")



if __name__ == '__main__':
    main()
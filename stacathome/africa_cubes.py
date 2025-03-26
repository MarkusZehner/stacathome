import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='/Net/Groups/BGI/work_1/scratch/DeepCube/sampled_minicubes.csv')
    parser.add_argument('--outfile', type=str, default='static/proto_dataset/africa_points.csv')
    args = parser.parse_args()

    csv = pd.read_csv(args.infile)

    lat_offset = (csv.MaxLat.values[0] - csv.MinLat.values[0]) / 2
    lon_offset = (csv.MaxLon.values[0] - csv.MinLon.values[0]) / 2
    lats = csv.MinLat + lat_offset
    lons = csv.MinLon + lon_offset
    positions = np.stack([lats, lons], axis=1)

    pd.DataFrame(positions, columns=['lat', 'lon']).to_csv(args.outfile, index=False)


if __name__ == '__main__':
    main()

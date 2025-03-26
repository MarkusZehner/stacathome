import sys
import os
import json

notebook_dir = "/Net/Groups/BGI/scratch/mzehner/code/stacathome/"
sys.path.append(notebook_dir)

from stacathome.sentinel_3_utils import load_s3_cube, cut_s3_cubes
from stacathome.request import probe_request
from stacathome.utils import parse_dec_to_lon_lat_point, parse_dms_to_lon_lat_point
from pystac import Item
from shapely.geometry import Point

if __name__ == '__main__':
    collection = "sentinel-3-synergy-syn-l2-netcdf"
    edge_length_m = 10000
    target_res_m = 300
    edge_length_pix = (edge_length_m // target_res_m) + 1
    flux_test_pos = [
        ('AU-Dry', '-15.2588, 132.3706'),
        ('AU-How', '-12.4943, 131.1523'),
        ('BE-Lon', '50.5516, 4.7462'),
        ('CD-Ygb', '0.8144, 24.5025'),
        ('CH-Dav', '46.8153, 9.8559'),
        ('CZ-Lnz', '48.6816, 16.9464'),
        ('DE-Hai', '51.0792, 10.4530'),
        ('DE-RuR', '50.6219, 6.3041'),
        ('DE-Tha', '50.9626, 13.5651'),
        ('ES-LMa', '39.9415, -5.7734'),
        ('FR-Fon', '48.4764, 2.7801'),
        ('GF-Guy', '5.2788, -52.9249'),
        ('IT-Noe', '40.6062, 8.1512'),
        ('US-Rpf', '65.1198, -147.4290'),
        ('US-SRG', '31.7894, -110.8277'),
        ('US-Tw4', '38.1027, -121.6413'),
        ('US-UMB', '45.5598, -84.7138'),
        ('US-UMd', '45.5625, -84.6975'),
        ('US-Var', '38.4133, -120.9507'),
        ('US-xDS', '28.1250, -81.4362'),
    ]

    workdir = '/Net/Groups/BGI/work_4/scratch/jnelson/Sen3_cutouts'

    for fname, position_str in flux_test_pos:
        try:
            position = parse_dms_to_lon_lat_point(position_str)
        except TypeError:
            pass
        try:
            position = parse_dec_to_lon_lat_point(position_str)
        except (TypeError, ValueError):
            pass
        assert position is not None
        assert isinstance(position, Point)

        print(f"Processing {fname} at {position}", flush=True)

        json_file = os.path.join(workdir, fname, f"{fname}_S3_query.json")
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                query_dict = json.load(f)

            query_dict = [Item.from_dict(feature) for feature in query_dict["features"]]
            print(f"Loaded {len(query_dict)} scenes", flush=True)
        else:
            print(f"Querying {fname}", flush=True)
            probe_dict = probe_request(
                point_wgs84=position, distance_in_m=edge_length_m, collection=[collection], return_box=True, limit=9999
            )
            print(f"Found {len(probe_dict[collection][0])} scenes", flush=True)
            query_dict = load_s3_cube(probe_dict, fname, workdir)

        cut_s3_cubes(
            query_dict,
            fname,
            position,
            edge_length_pix,
            workdir,
        )
        print(f"Finished {fname}", flush=True)

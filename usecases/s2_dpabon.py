import sys
import os
# import geopandas as gpd
# import numpy as np

notebook_dir = "/Net/Groups/BGI/scratch/mzehner/code/stacathome/"
sys.path.append(notebook_dir)

from stacathome.ymca import get_minicube
from stacathome.asset_specs import get_attributes

import pandas as pd

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate EU Minicubes')
    parser.add_argument('--start', type=int, required=True, help='Start index')
    parser.add_argument('--end', type=int, required=True, help='End index')
    args = parser.parse_args()
    start_index = args.start
    end_index = args.end

    file_path = os.path.join(notebook_dir, '/Net/Groups/BGI/scratch/mzehner/code/stacathome/usecases/danielPabon_missing_sites_post_2015.csv')
    df_sites = pd.read_csv(file_path, usecols=['site', 'lat', 'lon'])

    lat_lon_strings = []
    for i in df_sites.iterrows():
        lat_lon_strings.append([i[1].site , f'{i[1].lat}, {i[1].lon}'])

    edge_length_m = 10000
    target_res_m = 20

    workdir = '/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/dpabon'
    sel_bands = {'sentinel-2-l2a': list(get_attributes('sentinel-2-l2a')['data_attrs']['Band']),
                 # 'modis-13Q1-061': {'250m_16_days_NDVI', '250m_16_days_EVI', '250m_16_days_VI_Quality'},
                 # 'esa-worldcover': {'map'}
                 }
    time_range = {'sentinel-2-l2a': (2014, 2025),
                  # 'modis-13Q1-061': (2000, 2025),
                  # 'esa-worldcover': (2021)
                  }
    get_minicube(lat_lon_strings[start_index:end_index], time_range=time_range, edge_length_m=edge_length_m,
                 target_res_m=target_res_m, sel_bands=sel_bands, workdir=workdir)

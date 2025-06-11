from pathlib import Path
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import sys
import pickle
import geopandas as gpd
notebook_dir = "/Net/Groups/BGI/scratch/mzehner/code/stacathome/"
sys.path.append(notebook_dir)

from stacathome.redo_classes.requests import stacathome_wrapper
from stacathome.utils import run_with_multiprocessing


def process_tile(args):
    i, area, time_range, tmp_path_out, final_path_out, collections, subset_bands = list(args.values())

    for collection in collections:
        name_ident = f'{i}_{time_range}'
        file_path = tmp_path_out / f'{collection}_{name_ident}.zarr.zip'
        file_name = file_path.name
        lock_file = tmp_path_out / f"{file_name}.lock"
        final_path = final_path_out / file_name

        if os.path.exists(final_path):
            print(f"File {final_path} already exists, skipping.")
            return

        try:
            lock_file.touch(exist_ok=False)
        except FileExistsError:
            print(f"Lock exists for {file_name}, skipping.")
            return

        try:
            run_with_multiprocessing(
                stacathome_wrapper,
                area=area,
                time_range=time_range,
                collections=collection,
                subset_bands=subset_bands[collection],
                path_out=tmp_path_out,
                name_ident=name_ident,
            )

            shutil.move(str(file_path), final_path)
            print(f"Moved to {final_path}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
        finally:
            if lock_file.exists():
                lock_file.unlink()


if __name__ == "__main__":
    print('start')
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    print(f"Using {num_workers} workers.")

    collections = [
        # 'sentinel-2-l2a',
        'sentinel-1-rtc',
    ]
    subset_bands = {
        # 'sentinel-2-l2a': [],
        'sentinel-1-rtc' : [],
    }
    time_ranges = {
        # 'sentinel-2-l2a': '',
        'sentinel-1-rtc': '',
    }

    tiles_and_times = pickle.load(open('/Net/Groups/BGI/work_5/scratch/mzehner/s2_cubes_to_request_thuringia_2.pkl', 'rb'))
    final_path_out = Path('/Net/Groups/BGI/work_5/scratch/mzehner/S1thuringia')
    tmp_path_out = Path('/Net/Groups/BGI/work_5/scratch/mzehner/S1thuringia/tmp')

    os.makedirs(tmp_path_out, exist_ok=True)

    buckets_th = gpd.read_parquet('/Net/Groups/BGI/work_5/scratch/mzehner/buckets_with_forest_g_35p_time.parquet')

    args_list = [{
        'i' : i[0],
        'area' : buckets_th.where(buckets_th['tile'] == i[0]).dropna().geometry.iloc[0],
        'time_range' : str(i[1]),
        'tmp_path_out': tmp_path_out,
        'final_path_out': final_path_out,
        'collections': collections,
        'subset_bands': subset_bands} for i in tiles_and_times]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_tile, args_list)

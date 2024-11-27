from urllib.request import urlretrieve
from os import makedirs, remove as os_remove
from os import path as os_path

from dask.distributed import fire_and_forget
from planetary_computer import sign as pc_sign

def get_asset(href, save_path):
    makedirs(os_path.dirname(save_path), exist_ok=True)
    try:
        urlretrieve(pc_sign(href), save_path)
    except (KeyboardInterrupt, SystemExit):
        if os_path.exists(save_path):
            try:
                os_remove(save_path)
            except Exception as e:
                print(f"Error during cleanup of file {save_path}:", e)
    except Exception as e:
        print(f"Error downloading {href}:", e)
    return None



def download_item(item):
    if isinstance(item, tuple):
        get_asset(*item)
    if isinstance(item, list):
        for i in item:
            get_asset(*i)
    return None


def parallel_download(items, client):
    """
    Download items in parallel.

    Parameters
    ----------
    items: list
        The items to download.
    """
    do = []
    for i in items:
        do.append(client.submit(download_item, i))
        # do.append(dask.delayed(download_item)(i))
    # downloads = db.from_sequence(items).map(download_item)
    fire_and_forget(do)
    # dask.compute(*do)
    return None

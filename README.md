# Yet another STAC-downloader

## install:
```bash
# Using conda
conda/mamba env create -n my-package-env -f environment.yaml

conda/mamba activate my-package-env

uv sync

# for dev
uv sync --all-extras --all-groups --cache-dir /some/dir/on/the/same/disk/as/project
```

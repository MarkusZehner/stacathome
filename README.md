# Yet another STAC-downloader

## Benefits:
### Two options for data access:
- Wall-To-Wall downloading of S2
  - Requesting, downloading and cataloging the full tiles has greater data throughput compared to on-the-fly datacube generation from STAC catalogs
- Sparse sampling
  - sentinel-2 grid aware (few resampling operations as possible) requests either at native or selected resolution into zarr or zipped zarr containers

### other features:
- temporal resampling with either image with maximum count of valid SCL values or ranked iterative fill





![image](https://github.com/user-attachments/assets/1061bce8-a2d3-4b45-bc1d-7919fc7f8e39)

import numpy as np
import xarray as xr

def temporal_resample_valid_steps(data, SCL_Valid = None, advanced=True):
    # using the SCL layer here, below values are 'valid' after 
    # L. Baetens, C. Desjardins, and O. Hagolle, “Validation of 
    # Copernicus Sentinel-2 cloud masks obtained from MAJA, Sen2Cor, 
    # and FMask processors using reference cloud masks generated with 
    # a supervised active learning procedure,” 
    # Remote Sensing, vol. 11, no. 4, p. 433, 2019.

    if SCL_Valid is None:
        SCL_Valid = [4, 5, 6, 7, 11]

    # mask and sum up most valid steps
    scl_masks = xr.where(data.SCL.isin([2, 4, 5, 6, 7, 11]), 1, 0)
    valid_sum = scl_masks.sum(dim=['x', 'y']).values
    most_valid = np.argsort(valid_sum)[::-1]

    # fast: just return this step
    resampled = data.isel(time=most_valid[0])

    # more valid data: 
    # set all masked out pixels to nan, iterate from most to least valid step and fill the pixels
    if advanced:
        time = resampled.time  # needs to be added as this dim gets lost here
        resampled = xr.where(scl_masks.isel(time=most_valid[0])==1, resampled, np.nan)
        for i in range(1, len(most_valid)):
            add = xr.where(scl_masks.isel(time=most_valid[i])==1, data.isel(time=most_valid[i]), np.nan)

            resampled = xr.where(resampled > 0, resampled, add)
        resampled.coords['time'] = time
    return resampled

# Example usage
# time_range = reference_cube.time_range
# time_period = reference_cube.time_period
# ds = no_group.sel(time=slice('2016-01-01', time_range[1]))
# resampler = ds.resample(skipna=True, time=time_period) #, origin='start')
# resampled_ds = resampler.apply(temporal_resample_valid_steps)
# time_bins = pd.date_range(start=time_range[0], end=time_range[1], freq='5D')
# time_labels = time_bins.values + np.timedelta64(60, 'h')

# resampled_ds['time'] = time_labels
import json
import os
import time
from datetime import datetime
import warnings
import sys

import numba as nb
import numpy as np
import xarray as xr
from numba import njit, prange
from numpy.linalg import LinAlgError, pinv, solve
from odc.geo.geobox import GeoBox
from odc.stac import load
from pyproj import CRS, Transformer
from pystac import Item
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

notebook_dir = "/Net/Groups/BGI/scratch/mzehner/code/stacathome/"
sys.path.append(notebook_dir)
from stacathome.asset_specs import get_attributes, get_band_attributes_s2, get_resampling_per_band


def find_nearest_indices(high_res_coords, low_res_coords):
    """Find nearest neighbors in low-resolution dataset for each point in high-resolution dataset."""
    tree = cKDTree(low_res_coords)
    _, indices = tree.query(high_res_coords)
    return indices


def load_subset_otf_from_tiffs(dev_path, bands, collection, tile, i_y, i_x, c_size, time_range, return_box=False):
    dev_path = "/Net/Groups/BGI/work_5/scratch/Somalia_VCI_test" if dev_path is None else dev_path

    os.makedirs(dev_path, exist_ok=True)

    t = '37NHE' if tile is None else tile
    bands = ['B02', 'B03', 'B04', 'B08', 'SCL'] if bands is None else bands
    collection = 'sentinel-2-l2a' if collection is None else collection
    c_size = 512 if c_size is None else c_size

    query_path = os.path.join(dev_path, t, 'sentinel-2-l2a_queries')
    data_path = os.path.join(dev_path, t, 'sentinel-2-l2a_data')
    os.listdir(query_path)
    with open(os.path.join(query_path, os.listdir(query_path)[0])) as json_file:
        query_dict = json.load(json_file)
    query = [Item.from_dict(feature) for feature in query_dict["features"]]

    def replace_href(values, local_path):
        values.href = os.path.join(local_path, values.href.split('/')[-1])
        return values

    asset_keys = query[0].assets.keys()
    for q in range(len(query)):
        query[q].assets = {
            k: replace_href(v, data_path)
            for k, v in query[q].assets.items()
            if (k in asset_keys and os.path.exists(os.path.join(data_path, v.href.split('/')[-1])))
        }

    if time_range is not None:
        query = [
            q
            for q in query
            if time_range[0] < q.properties['datetime'][:-17] and q.properties['datetime'][:-17] < time_range[1]
        ]

    bbox = query[0].assets['B02'].extra_fields['proj:bbox']
    crs = query[0].properties['proj:epsg']
    request_geobox = GeoBox.from_bbox(bbox, CRS.from_epsg(crs), resolution=10)

    request_geobox_sub = request_geobox[i_y * c_size : i_y * c_size + c_size, i_x * c_size : i_x * c_size + 512]

    s2_attributes = get_attributes(collection)['data_attrs']
    _s2_dytpes = dict(zip(s2_attributes["Band"], s2_attributes["Data Type"]))

    resampling = get_resampling_per_band(abs(int(request_geobox.resolution.x)), bands=bands, collection=collection)

    otf_cube = load(
        query,
        bands=bands,
        chunks={'time': -1, 'x': 512, 'y': 512},
        geobox=request_geobox_sub,
        dtype=_s2_dytpes,
        resampling=resampling,
        groupby="solar_day",
    )

    attributes = get_attributes(collection)
    data_atts = attributes.pop("data_attrs")
    band_attrs = get_band_attributes_s2(data_atts, otf_cube.data_vars.keys())
    for band in band_attrs.keys():
        otf_cube[band].attrs = band_attrs[band]
        if band == 'SCL':
            otf_cube['SCL'].attrs = attributes['SCL']

    if return_box:
        return otf_cube, request_geobox_sub
    return otf_cube


def harmonize_bands(B02, B03, B04, B08, SCL, Time):
    # Convert to float to avoid integer casting issues
    B02, B03, B04, B08 = (band.astype(np.float32) for band in [B02, B03, B04, B08])
    Time = Time.copy()

    threshold_date = np.datetime64('2022-01-25T00:00:00')
    basline_mask = Time > threshold_date
    scl_mask = np.isin(SCL, [2, 4, 5, 6, 7, 11])
    offset = 1000

    for band in [B02, B03, B04, B08]:
        band[basline_mask] -= offset
        band /= 10000.0  # Ensure floating point division

    B02, B03, B04, B08 = (np.where(scl_mask, band, np.nan) for band in [B02, B03, B04, B08])

    extra = 2 * B02 - 0.95 * B03 - 0.05
    B04, B08 = (np.where(extra <= 0, band, np.nan) for band in [B04, B08])

    return B04, B08, Time.astype('datetime64[D]')


def compute_ndvi(B04, B08):
    ndvi = (B08 - B04) / (B08 + B04).clip(1e-8, None)
    return ndvi


def resample_weekly_custom_daterange(da, Time, time_range, t_delta=(7, "D")):
    out = np.full(time_range.shape, np.nan)
    for i, date in enumerate(time_range):
        mask = (Time >= date) & (Time < date + np.timedelta64(*t_delta))
        if np.any(mask):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out[i] = np.nanmean(da[mask])
    return out


def weekly_ndvi(ndvi, Time, t_delta=(7, "D")):
    date_range = define_time_range(Time)
    ndvi_weekly_values = np.full(date_range.shape, np.nan)

    for i, date in enumerate(date_range):
        mask = (Time >= date) & (Time < date + np.timedelta64(*t_delta))
        if np.any(mask):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ndvi_weekly_values[i] = np.nanmax(ndvi[mask])

    valid_mask = ~np.isnan(ndvi_weekly_values)
    if np.sum(valid_mask) >= ndvi_weekly_values.shape[0] / 10:
        pchip_interpolator = PchipInterpolator(date_range[valid_mask].astype(float),
                                               ndvi_weekly_values[valid_mask], extrapolate=False)
        ndvi_pchip_values = pchip_interpolator(date_range.astype(float))
    else:
        ndvi_pchip_values = np.full_like(ndvi_weekly_values, np.nan)

    return date_range + np.timedelta64(t_delta[0] - 1, t_delta[1]), ndvi_pchip_values


def daily_ndvi(ndvi, Time, start_day="2016-01-01"):
    date_range = np.arange(np.datetime64(start_day), Time[-1] + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    ndvi_daily_values = np.full(date_range.shape, np.nan)
    unique_time, unique_indices = np.unique(Time, return_index=True)
    indices = np.searchsorted(date_range, unique_time)
    ndvi_daily_values[indices] = ndvi[unique_indices]

    valid_mask = ~np.isnan(ndvi_daily_values)
    if np.any(valid_mask):
        pchip_interpolator = PchipInterpolator(date_range[valid_mask].astype(float),
                                               ndvi_daily_values[valid_mask], extrapolate=False)
        ndvi_pchip_values = pchip_interpolator(date_range.astype(float))
    else:
        ndvi_pchip_values = np.full_like(ndvi_daily_values, np.nan)

    nan_mask = np.isnan(ndvi_pchip_values)
    nearest_interpolated_values = np.interp(
        date_range.astype(float), date_range[~nan_mask].astype(float), ndvi_pchip_values[~nan_mask]
    )

    return date_range, nearest_interpolated_values


@njit(parallel=True)
def compute_ndvi_min_max(ndvi_values, day_of_year):
    ndvi_min = np.full(366, np.nan)
    ndvi_max = np.full(366, np.nan)
    for doy in prange(366):
        mask = day_of_year == doy
        if np.any(mask):
            ndvi_min[doy] = np.nanmin(ndvi_values[mask])
            ndvi_max[doy] = np.nanmax(ndvi_values[mask])
    return ndvi_min, ndvi_max


@njit(parallel=True)
def compute_ndvi_min_max_smoothed(ndvi_values, doy_index, window_size=12):
    len_ar = np.unique(doy_index).shape[0]  # Number of unique DOY values
    ndvi_min = np.full(len_ar, np.nan, dtype=np.float32)
    ndvi_max = np.full(len_ar, np.nan, dtype=np.float32)

    # Compute min/max NDVI per DOY
    for i in prange(len_ar):
        min_val = np.inf
        max_val = -np.inf
        for j in range(ndvi_values.shape[0]):  # Avoiding boolean masks for speed
            if doy_index[j] == doy_index[i]:
                val = ndvi_values[j]
                if not np.isnan(val):
                    if val < min_val:
                        min_val = val
                    if val > max_val:
                        max_val = val

        if min_val != np.inf:
            ndvi_min[i] = min_val
        if max_val != -np.inf:
            ndvi_max[i] = max_val

    # Apply a circular (rolling) smoothing filter
    ndvi_min_smoothed = np.full(len_ar, np.nan, dtype=np.float32)
    ndvi_max_smoothed = np.full(len_ar, np.nan, dtype=np.float32)

    for doy in prange(len_ar):
        # Define the rolling window with wrap-around
        start = doy - window_size // 2
        end = doy + window_size // 2 + 1
        indices = (np.arange(start, end) % len_ar).astype(np.int32)

        # Compute mean, ignoring NaNs
        sum_min, count_min = 0.0, 0
        sum_max, count_max = 0.0, 0

        for idx in indices:
            if not np.isnan(ndvi_min[idx]):
                sum_min += ndvi_min[idx]
                count_min += 1
            if not np.isnan(ndvi_max[idx]):
                sum_max += ndvi_max[idx]
                count_max += 1

        if count_min > 0:
            ndvi_min_smoothed[doy] = sum_min / count_min
        if count_max > 0:
            ndvi_max_smoothed[doy] = sum_max / count_max

    return ndvi_min_smoothed, ndvi_max_smoothed


def compute_vci_weekly(ndvi_pchip_values, date_range, smooth_window_extremes=12):
    # Convert dates to day of the year
    doy_index = date_range.astype('datetime64[D]') - date_range.astype('datetime64[Y]') + 1
    # fix overlapping into new year
    doy_index[doy_index < 7] = 371

    # Compute min/max NDVI based on 7-day intervals
    ndvi_min, ndvi_max = compute_ndvi_min_max_smoothed(ndvi_pchip_values,
                                                       doy_index,
                                                       window_size=smooth_window_extremes)

    # Apply VCI calculation
    temp_ndvi_min = ndvi_min[(doy_index.astype(int) - 1) // 7]
    temp_ndvi_max = ndvi_max[(doy_index.astype(int) - 1) // 7]

    return (ndvi_pchip_values - temp_ndvi_min) / (temp_ndvi_max - temp_ndvi_min).clip(0.01, None)


def compute_vci_doy(ndvi_pchip_values, date_range, start_day="2016-01-01"):
    day_of_year = ((date_range - np.datetime64(start_day)) % np.timedelta64(365, "D")).astype(int)
    ndvi_min, ndvi_max = compute_ndvi_min_max(ndvi_pchip_values, day_of_year)
    temp_ndvi_min = ndvi_min[day_of_year]
    temp_ndvi_max = ndvi_max[day_of_year]
    return (ndvi_pchip_values - temp_ndvi_min) / (temp_ndvi_max - temp_ndvi_min).clip(0.01, None)


@njit
def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        start = max(0, i - window // 2)
        end = min(len(arr), i + window // 2 + 1)
        result[i] = np.nanmean(arr[start:end])
    return result


@njit
def weekly_mean(arr, offset, step=7):
    # Adjust the array to start from the first complete week
    adjusted_arr = arr[offset:]

    # Calculate the number of weeks
    n_weeks = (len(adjusted_arr) + step - 1) // step  # Ensures last week is included
    weekly_arr = np.full(n_weeks, np.nan)

    for i in range(n_weeks):
        start = i * step
        end = min(len(adjusted_arr), (i + 1) * step)  # Avoids index overflow
        weekly_arr[i] = np.nanmean(adjusted_arr[start:end])

    return weekly_arr


def calculate_offset(start_date, step=7):
    # Convert numpy datetime64 to Python datetime
    start_date = start_date.astype('M8[D]').astype(datetime)

    # Calculate the offset to align with the start_date
    start_day_of_week = start_date.weekday()  # Monday is 0 and Sunday is 6
    offset = (step - start_day_of_week) % step  # Days to the next full week
    return offset


def vci_weekly(B02, B03, B04, B08, SCL, Time, smooth_window_extremes=12):
    B04, B08, Time = harmonize_bands(B02, B03, B04, B08, SCL, Time)
    ndvi = compute_ndvi(B04, B08)
    date_range, ndvi_pchip_values = weekly_ndvi(ndvi, Time)
    vci = compute_vci_weekly(ndvi_pchip_values, date_range, smooth_window_extremes=smooth_window_extremes)
    vci3m = rolling_mean(vci, 12)
    # offset = calculate_offset(Time[0])
    # weekly_vci3m = weekly_mean(vci3m, offset=0)

    return vci3m


def define_time_range(Time):
    start_years = np.unique(Time.astype('datetime64[Y]'))
    date_range = np.concatenate([
        np.arange(year,
                  # year + np.timedelta64(366 if (year.astype(int) % 4 == 0 and (year.astype(int) % 100 != 0 or year.astype(int) % 400 == 0)) else 365, "D"),
                  year + np.timedelta64(365 - 7, "D"),
                  np.timedelta64(7, "D"))
        for year in start_years
    ])
    return date_range


def get_vci3m_weekly(dataset, smooth_window_extremes=12):
    t_ax = define_time_range(dataset.time.values).astype("datetime64[ns]") + np.timedelta64(6, "D")

    result = xr.apply_ufunc(
        vci_weekly,
        dataset.B02,
        dataset.B03,
        dataset.B04,
        dataset.B08,
        dataset.SCL,
        dataset.time,
        kwargs={
            'smooth_window_extremes': smooth_window_extremes,
        },
        input_core_dims=[['time'], ['time'], ['time'], ['time'], ['time'], ['time']],
        output_core_dims=[['weeks']],
        dask_gufunc_kwargs={'output_sizes': {'weeks': len(t_ax)}},
        dask='parallelized',
        output_dtypes=[np.float32],
        vectorize=True,
    )

    result = result.assign_coords(weeks=t_ax)
    result = result.rename({'weeks': 'time'})
    return result.to_dataset(name='vci3m_weekly')


def __get_buffer_shift(shifts: np.array) -> int:
    """
    Calculates the buffer shift required for the shifts.
    If shifts go over 0, the range is needed to be buffered, else the max absolute shift is used.

    Parameters
    ----------
    shifts : np.array
        The shifts to apply.

    Returns
    -------
    int
        The buffer shift.
    """
    if max(shifts) <= 0 or min(shifts) >= 0:
        return max(abs(shifts))
    return max(shifts) + abs(min(shifts))


def __get_array_indicators_for_lin_reg(
    shifts: np.array,
    in_dims: int,
    upper_bound_input_shift: int = 0,
    predict_on_variable_pos_indicator: list = None,
) -> tuple[np.array, np.array]:
    """
    Get the array indicators for the linear regression model.
    The array is organized so that the first n_features belong to the first shift,
    the second n_features to the second shift, and so on.
    With n_features being the number of features (second dim) in X.

    Parameters
    ----------
    shifts : np.array
        The shifts to apply.
    in_dims : int
        The number of features in the input data.
    upper_bound_input_shift : int
        The upper bound of shifts for what will be considered as input data.
    predict_on_variable_pos_indicator : list
        The list of positions of the features to predict on. If None, the first feature is predicted on.

    Returns
    -------
    (np.array, np.array)
        The array indicators with shape (n_features*len(shifts)) for the backward shifts and the forward shifts.
    """
    if not predict_on_variable_pos_indicator:
        predict_on_variable_pos_indicator = [0]
    # This gets the positions of all backshifts (assumed used for input X)
    shifts_backwards = (np.array(shifts <= upper_bound_input_shift)).repeat(in_dims)
    # repeat the shifts for the number of features, as is sorted by add_shift_values_nb
    predict_on = (np.array(shifts > upper_bound_input_shift)).repeat(in_dims)
    # predict on the first feature only where the shifts are positive
    predict_on = np.full((in_dims), fill_value=False)
    # we want to predict on the first feature only
    predict_on[predict_on_variable_pos_indicator] = True
    # repeat the pattern to get the first feature for all shifts
    predict_on = np.tile(predict_on, shifts.shape[0])
    # we don't want to predict on the first feature for the backwards shifts
    predict_on[: shifts_backwards.sum()] = False
    return (shifts_backwards, predict_on)


@nb.jit(nopython=True)
def add_shift_values_nb(X: np.ndarray, shifts: np.array, buffer: int) -> np.ndarray:
    """
    Shifts the data by given values and collects the shifted values in a new array.
    The array is buffered and clipped within to only contain valid rows without 0-fills.
    The resulting array is organized so that the first n_features belong to the first shift,
    the second n_features to the second shift, and so on.
    With n_features being the number of features (second dim) in X.

    Parameters
    ----------
    X : np.ndarray
        The input data with shape (n_samples, n_features).
    shifts : np.array
        The shifts to apply.
    buffer : int
        The buffer size required for the shifts. Calculated by __get_buffer_shift.

    Returns
    -------
    np.ndarray
        The shifted data with shape (n_samples-2*buffer_shift, n_features*len(shifts)).
    """
    X_buf = np.zeros((X.shape[0] + 2 * buffer, X.shape[1]))
    X_buf[buffer:-buffer] = X
    m, n = X_buf.shape
    out = np.zeros((X_buf.shape[0], X_buf.shape[1] * len(shifts)))
    for i in nb.prange(len(shifts)):
        start_shift = buffer - shifts[i]

        end_shift = start_shift + (m - 2 * buffer)

        out[buffer:-buffer, i * n : (i + 1) * n] = X_buf[start_shift:end_shift, :]

    return out[buffer * 2 : -buffer * 2]


@nb.jit(nopython=True)
def __check_nan(X: np.ndarray) -> np.ndarray:
    """
    Check if there are any NaNs in the rows of the input array.

    Parameters
    ----------
    X : np.ndarray
        Input array to check for NaNs.

    Returns
    -------
    np.ndarray
        Boolean array with True if there are no NaNs in the row, False otherwise.
    """
    out = np.full(X.shape[0], dtype=np.bool_, fill_value=True)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X[i, j]):
                out[i] = False
                break
    return out


# @nb.jit(nopython=True, parallel=True)
# def moving_ols_slope_predict_baseline(X: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
#     """
#     Calculate the moving OLS without intercept for a multivariate regression using numba.

#     Parameters
#     ----------
#     X : np.ndarray
#         The input data with shape (n_samples, n_features).
#     y : np.ndarray
#         The target data with shape (n_samples, n_targets).
#     window : int
#         The window size.

#     Returns
#     -------
#     np.ndarray
#         The output predictions with shape (n_samples-window, n_targets).
#     """
#     out = np.zeros((X.shape[0] - (window), y.shape[1]))
#     for i in nb.prange(0, len(X) - window):
#         X_window = X[i : i + window].copy()
#         y_window = y[i : i + window].copy()
#         last = X[i + window].copy()
#         # get slope by inversion, switch to pinv(slower) if det == 0
#         XTX = X_window.T @ X_window
#         if np.linalg.det(XTX) == 0:
#             inv = np.linalg.pinv(XTX)
#         else:
#             inv = np.linalg.inv(XTX)
#         slope_coeff = inv @ (X_window.T @ y_window)
#         # multiply the last value in window of X with the slope to get the prediction
#         out[i] = slope_coeff.T @ last
#     return out


def prepare_moving(X, y):
    nv, nt = X.shape  # X: (n_features, n_samples)
    n_out = y.shape[0]  # Number of outputs
    X1 = np.vstack([X, y])  # Remove bias term for pure slope calculation
    R = np.zeros((nv + n_out, nv + n_out))
    SR = R.copy()
    LUR = np.zeros((nv, nv))  # Exclude bias term here
    RY = np.zeros((nv, n_out))  # Adjust for multiple outputs
    outar = np.full((n_out, nt), np.nan)  # Adjust output shape
    return X1, R, SR, LUR, RY, outar


def movinglinreg(X, y, window, prep=None):
    nv, nt = X.shape
    n_out = y.shape[0]  # Number of outputs
    if prep is None:
        prep = prepare_moving(X, y)

    outoffset = -(window - 1)  # Align to predict at the end of the window
    X1, R, SR, LUR, RY, outar = prep

    # Initialize with the first window
    XV = X1[:-n_out, :window]  # Exclude y for initial R
    YV = X1[-n_out:, :window]  # Extract all y values
    R[:nv, :nv] = XV @ XV.T  # In-place matrix multiplication
    R[:nv, nv : nv + n_out] = XV @ YV.T  # Cross terms

    predictions = np.full((n_out, nt), np.nan)

    for i in range(window, nt + 1):
        # Copy R into SR (symmetric matrix)
        np.copyto(SR, R)

        # Correct RY indexing to pull y cross terms
        np.copyto(RY, SR[:nv, nv : nv + n_out])
        np.copyto(LUR, SR[:nv, :nv])

        # Estimate coefficients using inv or pinv
        try:
            coeffs = solve(LUR, RY)
        except LinAlgError:
            coeffs = pinv(LUR) @ RY

        # Compute predicted y values using only the slopes
        x_current = X[:, i - 1]  # Use the last point in the current window
        predictions[:, i + outoffset] = coeffs.T @ x_current

        # Update R in-place for the sliding window
        if i < nt:
            nextval = X1[:-n_out, i].reshape(-1, 1)  # Exclude y
            lastval = X1[:-n_out, i - window].reshape(-1, 1)
            R[:nv, :nv] += nextval @ nextval.T
            R[:nv, :nv] -= lastval @ lastval.T

            # Update cross terms with y
            next_y = X1[-n_out:, i].reshape(-1, 1)
            last_y = X1[-n_out:, i - window].reshape(-1, 1)
            R[:nv, nv : nv + n_out] += nextval @ next_y.T
            R[:nv, nv : nv + n_out] -= lastval @ last_y.T

    return predictions.T[1 : -(window - 1)]


def rolling_linreg(data, weather, window, shifts, buffer, shifts_backwards, predict_on):
    """
    Xarray wrapper for the moving_ols_slope_predict function.
    Data is shifted against itself for window of lagged input and prediction dates.
    Shifts <= 0 are handled as input variables, shifts > 0 are handled as prediction variables.
    Use __get_array_indicators_for_lin_reg to get the shifts_backwards and predict_on arrays.

    Parameters
    ----------
    data : np.ndarray
        The input VCI3M data with shape (1, time).
    weather : np.ndarray
        The input weather data with shape (n_features, time).
    window : int
        The window size.
    shifts : np.array
        The shifts to apply for input and prediction.
    buffer : int
        The buffer size required for the shifts. Calculated by __get_buffer_shift.
    in_dims : int
        The number of features in the input data.
    shifts_backwards : np.array
        Boolean array to select the columns from the combined input as variables.
        see __get_array_indicators_for_lin_reg
    predict_on : np.array
        Boolean array to select the columns from the combined input as targets.
        see __get_array_indicators_for_lin_reg

    Returns
    -------
    np.ndarray
        The predicted values with shape (time, n_targets).
    """
    # Add satellite index (1, time) to the weather data (n_features, time)
    combine = np.concatenate([data.reshape(1, -1), weather], axis=0).T

    # data is shifted against itself for window of lagged input and prediction dates
    Xy = add_shift_values_nb(combine, shifts, buffer)

    # remove rows with nan, remember positions
    remove_nan = __check_nan(Xy)

    # create ouput arr, calculate results or give back empty if data is too short for window.
    return_arr = np.full((data.shape[0], predict_on.sum()), fill_value=np.nan)
    if remove_nan.sum() > window:
        Xy_ = Xy[remove_nan]
        out = np.full((Xy_.shape[0], predict_on.sum()), fill_value=np.nan)
        # out[window:] = moving_ols_slope_predict_baseline(
        #    Xy_[:, shifts_backwards], Xy_[:, predict_on], window
        # )
        out[window:] = movinglinreg(Xy_[:, shifts_backwards].T, Xy_[:, predict_on].T, window)
        buffer_nan = np.array([False]).repeat(buffer)
        pad_remove_nan = np.concatenate((buffer_nan, remove_nan, buffer_nan))
        return_arr[pad_remove_nan] = out
    return return_arr


if __name__ == '__main__':
    # get data from MPC
    # import argparse

    # from stacathome.request import request_data_by_tile
    # from stacathome.download import download_assets_parallel

    # tile = ['37NHE']
    # dev_path = "/Net/Groups/BGI/work_5/scratch/Somalia_VCI_test"
    # bands = ['B02', 'B03', 'B04', 'B08', 'SCL']

    # time_bins = ('2014-05-01', '2025-08-31')

    # to_process = []
    # for t in tile:
    #     query_path = os.path.join(dev_path, t, 'sentinel-2-l2a_queries')
    #     data_path = os.path.join(dev_path, t, 'sentinel-2-l2a_data')
    #     os.makedirs(query_path, exist_ok=True)
    #     os.makedirs(data_path, exist_ok=True)
    #     items = request_data_by_tile(
    #         tile_id=t,
    #         start_time=time_bins[0],
    #         end_time=time_bins[1],
    #         save_dir=query_path,
    #     )
    #     for i in items:
    #         for b in bands:
    #             to_process.append((i.assets[b].href,
    #                                os.path.join(data_path, i.assets[b].href.split('/')[-1])))

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--workers", type=int, default=int(os.getenv("SLURM_CPUS_PER_TASK", 4)), help="Number of workers")
    # args = parser.parse_args()

    # download_assets_parallel(to_process, max_workers=args.workers)

    import sys

    print('Starting at', time.ctime(), flush=True)
    i_y = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    i_x = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    t = time.perf_counter()

    print('Loading data at', time.ctime(), flush=True)
    era5_vars = ["pev", "t2m", "tp", "swvl1"]
    era5 = xr.open_zarr("/Net/Groups/BGI/tscratch/vbenson/EarthNet/droughtearthnet/data/Somalia/era5_0d25.zarr")

    # temp_offset_era5 = calculate_offset(era5.time.values[0])

    start_date = '2016-01-01'
    end_date = '2023-12-31'

    cube = load_subset_otf_from_tiffs(
        dev_path=None,
        bands=None,
        collection=None,
        tile=None,
        i_y=i_y,
        i_x=i_x,
        c_size=None,
        time_range=[start_date, end_date],
    )
    print(cube.time.values[0], cube.time.values[-1], flush=True)

    transformer = Transformer.from_crs(4326, cube.spatial_ref.attrs['crs_wkt'], always_xy=True)

    points_lr = [(transformer.transform(y, x)) for x in era5.lat.values for y in era5.lon.values]
    points_lr_back = [(y, x) for x in era5.lat.values for y in era5.lon.values]
    points_hr = [(y, x) for x in cube.x.values for y in cube.y.values]
    era5_index = [points_lr_back[i] for i in find_nearest_indices(points_hr, points_lr)]
    # assert len(np.unique(era5_index, axis=0))

    vci_3m_path = f'/Net/Groups/BGI/work_5/scratch/Somalia_VCI_test/37NHE/vci_results/weekly_vci3m_{i_y}_{i_x}.zarr'

    if not os.path.exists(vci_3m_path):
        print('Calculating VCI3M at', time.ctime(), flush=True)
        weekly_vci3m = get_vci3m_weekly(cube).load()

        print('Saving VCI3M at', time.ctime(), flush=True)
        weekly_vci3m.to_zarr(vci_3m_path, compute=True, mode='w')
    else:
        print('Loading VCI3M at', time.ctime(), flush=True)
        weekly_vci3m = xr.open_zarr(vci_3m_path)

    print('vci done')

    print(weekly_vci3m.time.values[0], weekly_vci3m.time.values[-1], flush=True)

    weekly_era5 = (
        era5.sel(time=slice(start_date, end_date))
        # .resample(time="1W", origin='start_day')
        # .mean()
        .sel(lat=era5_index[0][0], lon=era5_index[0][1], method="nearest")[era5_vars]
        .drop_vars(["lat", "lon"])
    ).load()

    weather_np = weekly_era5.to_array(dim='variable').values

    time_range = define_time_range(weekly_era5.time.values)
    era5_resampled = np.zeros((weather_np.shape[0], time_range.shape[0]))

    for i in range(weather_np.shape[0]):
        era5_resampled[i] = resample_weekly_custom_daterange(weather_np[i, :], weekly_era5.time.values, time_range)
    weekly_vci3m = weekly_vci3m.sel(time=slice(weekly_era5.time.values[0], weekly_era5.time.values[-1]))

    print(weekly_era5.time.values[0], weekly_era5.time.values[-1], flush=True)

    weekly_vci3m = weekly_vci3m.chunk(dict(time=-1, y='auto', x='auto'))

    shifts = np.array([-6, -5, -4, -3, -2, -1, 0, 4, 6, 8, 10, 12])
    buffer = __get_buffer_shift(shifts)
    in_dims = 1 + era5_resampled.shape[0]
    shifts_backwards, predict_on = __get_array_indicators_for_lin_reg(shifts, in_dims)

    window = 200

    print('Calculating VCI3M Forecast at', time.ctime(), flush=True)

    forecast = xr.apply_ufunc(
        rolling_linreg,
        weekly_vci3m,
        input_core_dims=[['time']],
        output_core_dims=[['time', 'shifts']],
        dask_gufunc_kwargs={'output_sizes': {'shifts': predict_on.sum()}},
        dask='parallelized',
        output_dtypes=[np.float32],
        vectorize=True,
        kwargs={
            'weather': era5_resampled,
            'window': window,
            'shifts': shifts,
            'buffer': buffer,
            'shifts_backwards': shifts_backwards,
            'predict_on': predict_on,
        },
    ).to_zarr(
        f'/Net/Groups/BGI/work_5/scratch/Somalia_VCI_test/37NHE/vci_results/subset_chunk_coords_{i_y}_{i_x}.zarr',
        compute=True,
        mode='w',
    )
    print('Finished at', time.ctime(), flush=True)
    print(f'elapsed time: {time.perf_counter() - t:.2f}', flush=True)

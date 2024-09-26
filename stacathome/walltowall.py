from os import listdir, path as os_path, makedirs
import copy
import warnings
from urllib.request import urlretrieve
from requests import get as requests_get
import numpy as np
from itertools import product


import pandas as pd
import geopandas as gpd
from json import load as json_load

from shapely import from_geojson
from shapely.ops import unary_union
from shapely.geometry import shape as s_shape

from dask import delayed


from planetary_computer import sign as pc_sign
from odc.stac import configure_rio
from pystac_client import Client
from pystac import ItemCollection
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros

from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from folium import Map, Popup, LatLngPopup, VegaLite, GeoJson, GeoJsonTooltip, CircleMarker, Circle
from altair import Chart, Axis, X as alt_X, Y as alt_Y, value as alt_value
import branca.colormap as cm


configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})


def get_countries_json(name):
    if check_if_country_in_repo(name):
        url = f"https://raw.githubusercontent.com/georgique/world-geojson/main/countries/{
            name}.json"
        r = requests_get(url)
        return r.json()


def check_if_country_in_repo(name):

    country_list = ['afghanistan', 'albania', 'algeria', 'andorra', 'angola',
                    'antigua_and_barbuda', 'argentina', 'armenia', 'australia',
                    'austria', 'azerbaijan', 'bahamas', 'bahrain', 'bangladesh',
                    'barbados', 'belarus', 'belgium', 'belize', 'benin', 'bhutan',
                    'bolivia', 'bosnia_and_herzegovina', 'botswana', 'brazil',
                    'brunei', 'bulgaria', 'burkina_faso', 'burundi', 'cambodia',
                    'cameroon', 'canada', 'cape_verde', 'central_african_republic',
                    'chad', 'chile', 'china', 'colombia', 'comoros', 'congo',
                    'cook_islands', 'costa_rica', 'croatia', 'cuba', 'cyprus',
                    'czech', 'democratic_congo', 'denmark', 'djibouti', 'dominica',
                    'dominican_republic', 'east_timor', 'ecuador', 'egypt',
                    'el_salvador', 'equatorial_guinea', 'eritrea', 'estonia',
                    'eswatini', 'ethiopia', 'fiji', 'finland', 'france', 'gabon',
                    'gambia', 'georgia', 'germany', 'ghana', 'greece', 'grenada',
                    'guatemala', 'guinea', 'guinea_bissau', 'guyana', 'haiti',
                    'honduras', 'hungary', 'iceland', 'india', 'indonesia', 'iran',
                    'iraq', 'ireland', 'israel', 'italy', 'ivory_coast', 'jamaica',
                    'japan', 'jordan', 'kazakhstan', 'kenya', 'kiribati', 'kuwait',
                    'kyrgyzstan', 'laos', 'latvia', 'lebanon', 'lesotho', 'liberia',
                    'libya', 'liechtenstein', 'lithuania', 'luxembourg', 'madagascar',
                    'malawi', 'malaysia', 'maldives', 'mali', 'malta', 'marshall_islands',
                    'mauritania', 'mauritius', 'mexico', 'micronesia', 'moldova', 'monaco',
                    'mongolia', 'montenegro', 'morocco', 'mozambique', 'myanmar', 'namibia',
                    'nauru', 'nepal', 'netherlands', 'new_zealand', 'nicaragua', 'niger',
                    'nigeria', 'niue', 'north_korea', 'north_macedonia', 'norway', 'oman',
                    'pakistan', 'palau', 'palestine', 'panama', 'papua_new_guinea', 'paraguay',
                    'peru', 'philippines', 'poland', 'portugal', 'qatar', 'romania', 'russia',
                    'rwanda', 'saint_kitts_and_nevis', 'saint_lucia', 'saint_vincent_and_the_grenadines',
                    'samoa', 'san_marino', 'sao_tome_and_principe', 'saudi_arabia', 'senegal',
                    'serbia', 'seychelles', 'sierra_leone', 'singapore', 'slovakia', 'slovenia',
                    'solomon_islands', 'somalia', 'south_africa', 'south_korea', 'south_sudan',
                    'spain', 'sri_lanka', 'sudan', 'suriname', 'sweden', 'switzerland', 'syria',
                    'tajikistan', 'tanzania', 'thailand', 'togo', 'tonga', 'trinidad_and_tobago',
                    'tunisia', 'turkey', 'turkmenistan', 'tuvalu', 'uganda', 'ukraine',
                    'united_arab_emirates', 'united_kingdom', 'uruguay', 'usa', 'uzbekistan',
                    'vanuatu', 'vatican', 'venezuela', 'vietnam', 'western_sahara', 'yemen',
                    'zambia', 'zimbabwe']
    if name.lower() in country_list:
        return True
    else:
        owner = "georgique"  # GitHub username or organization
        repo = "world-geojson"    # Repository name
        # Path within the repository (use an empty string if listing the root)
        path = "countries"
        # Branch name (e.g., 'main', 'master', or other branches)
        branch = "main"

        url = f"https://api.github.com/repos/{owner}/{
            repo}/contents/{path}?ref={branch}"

        # Make a GET request to the GitHub API
        response = requests_get(url)

        # Check if the request was successful
        if response.status_code == 200:
            contents = response.json()
            # List all filenames
            filenames = [item['name'].split(
                '.')[0] for item in contents if item['type'] == 'file']
            if name.lower() in filenames:
                return True
        else:
            print(f"Failed to retrieve data: {response.status_code}")

    possible_matches = [
        country for country in country_list if country.startswith(name[0])]
    print(f"Country {name} not found in the repository, maybe you meant one of these: {
          possible_matches}")
    return None


def bbox(lon_lat, resolution, xy_shape):
    # stolen from minicuber
    if resolution <= 1:
        raise UserWarning(
            "Resolution less than 1m! Did you input lat/lon instead of UTM?")

    utm_epsg = int(query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            lon_lat[0], lon_lat[1], lon_lat[0], lon_lat[1])
    )[0].code)

    transformer = Transformer.from_crs(4326, utm_epsg, always_xy=True)

    x_center, y_center = transformer.transform(*lon_lat)

    nx, ny = xy_shape

    x_left, x_right = x_center - resolution * \
        (nx//2), x_center + resolution * (nx//2)

    y_top, y_bottom = y_center + resolution * \
        (ny//2), y_center - resolution * (ny//2)

    # left, bottom, right, top
    return transformer.transform_bounds(x_left, y_bottom, x_right, y_top, direction='INVERSE')


# def simple_lat_lon_box(lon, lat, resolution, xy_shape):
#     """
#     Create a bounding box from a center point and resolution in latlon.
#     TODO get something that works better?
#     """

#     x_left, x_right = lon - resolution * \
#         (xy_shape[0]//2), lon + resolution * (xy_shape[0]//2)

#     y_top, y_bottom = lat + resolution * \
#         (xy_shape[1]//2), lat - resolution * (xy_shape[1]//2)


#     return x_left, y_bottom, x_right, y_top


def create_skeleton_zarr(geobox, zarr_path,
                         chunksize_xy=256,
                         time_chunk_size=400,
                         dtype=np.uint16, bands=None,
                         start_date=None, end_date=None, t_freq='1D',
                         overwrite=False):
    """
    Create a zarr store with the same structure as the datacube, but with no data.
    Time dimension is created with the given start_date, end_date and t_freq.
    This should accomodate differing timestamps in the cube chunks.

    Intended for S-2 data currently.

    Parameters
    ----------
    geobox: GeoBox
        The geobox of the data
    zarr_path: str
        The path to the zarr store
    chunksize_xy: int
        The size of the x and y chunks
    dtype: numpy.dtype
        The datatype of the data
    bands: list
        The bands to include in the zarr store
    start_date: str
        The start date of the time dimension
    end_date: str
        The end date of the time dimension
    t_freq: str
        The frequency of the time dimension

    """
    if not bands:
        bands = ['B02', 'B03', 'B04', 'B8A']
    assert set(bands) <= set(['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                              'B07', 'B08', 'B09', 'B11', 'B12', 'B8A', 'SCL', 'WVP'])
    xr_empty = xr_zeros(geobox, dtype=dtype, chunks=(time_chunk_size, chunksize_xy, chunksize_xy),
                        time=pd.date_range(start_date, end_date, freq=t_freq))
    xr_empty = xr_empty.to_dataset(name=bands[0])
    for band in bands:
        xr_empty[band] = xr_empty[bands[0]].astype(dtype)
        xr_empty[band].encoding['_FillValue'] = 0
    mode = 'w' if overwrite else 'w-'
    xr_empty.to_zarr(zarr_path, mode=mode,
                     write_empty_chunks=False, compute=False)


def get_geobox_from_aoi(aoi_shape, epsg, resolution, chunksize, return_aoi=False):
    if type(aoi_shape) == str:
        with open(aoi_shape) as f:
            d = json_load(f)
    else:
        d = aoi_shape
    aoi = from_geojson(f'{d['features'][0]['geometry']}'.replace("'", '"'))

    # buffer to adjust to chunksize
    bounds = aoi.buffer(resolution*chunksize/2).bounds
    geobox = GeoBox.from_bbox(
        bounds, crs=f"epsg:{epsg}", resolution=resolution)

    # slice to be a multiple of 256
    geobox = geobox[slice(geobox.shape[0] % chunksize//2, -(chunksize-geobox.shape[0] % chunksize//2)),
                    slice(geobox.shape[1] % chunksize//2, -(chunksize-geobox.shape[1] % chunksize//2))]
    if return_aoi:
        return geobox, aoi
    return geobox


def subset_geobox_by_chunk_nr(geobox, chunk_nr, chunksize_xy, chunk_table):
    chunk = chunk_table.loc[chunk_nr]
    ch_side = chunksize_xy//2
    index_slices = [chunk['lon_chunk']-ch_side, chunk['lon_chunk']+ch_side,
                    chunk['lat_chunk']-ch_side, chunk['lat_chunk']+ch_side]
    return geobox[index_slices[0]: index_slices[1],
                  index_slices[2]: index_slices[3]], index_slices


def subset_geobox_by_bbox_chunkwise(geobox, epsg, lat, lon, resolution_in_utm, subset_size, chunksize):
    """
    Gives back the defined area within the geobox rounded to chunks.
    """
    bbox_latlon = bbox((lat, lon), resolution_in_utm,
                       (subset_size, subset_size))
    overlap = geobox.overlap_roi(GeoBox.from_bbox(
        bbox_latlon, crs=f"epsg:{epsg}", resolution=geobox.resolution.x))
    index_slices = [(overlap[0].start//chunksize) * chunksize, (overlap[0].stop//chunksize+1) * chunksize,
                    (overlap[1].start//chunksize) * chunksize, (overlap[1].stop//chunksize+1) * chunksize]
    return geobox[index_slices[0]: index_slices[1],
                  index_slices[2]: index_slices[3]], index_slices


def check_request_against_local(items, out_path, requested_bands=None, report=True):
    """
    Check which of the requested assets are already downloaded and which are missing.
    list of available assets can be directly passed to odc-stac.
    """
    measurement_assets = ['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                          'B07', 'B08', 'B09', 'B11', 'B12', 'B8A', 'SCL', 'WVP']
    ingnore_assets = ['visual', 'preview', 'safe-manifest', 'granule-metadata',
                      'inspire-metadata', 'product-metadata', 'datastrip-metadata',
                      'tilejson', 'rendered_preview']

    if not requested_bands:
        requested_bands = measurement_assets
    else:
        assert set(requested_bands) <= set(measurement_assets)

        ingnore_assets = ingnore_assets + \
            [a for a in measurement_assets if a not in requested_bands]

    if type(items) == ItemCollection:
        local_items = items.clone()
        local_items = list(local_items)
    elif type(items) == list:
        local_items = copy.deepcopy(items)
    idx_to_pop = []
    not_downloaded_items = []
    to_download = []
    missing_assets = 0
    for i in range(len(local_items)):
        downloaded = True
        for b in requested_bands:
            try:
                # check if the file is already downloaded
                # if yes, add path to local_items
                save_path = os_path.join(*[out_path] +
                                         local_items[i].assets[b].href.split("?")[0].split("/")[-6:])
                if check_sentinel2_data_exists_with_min_size(save_path):
                    local_items[i].assets[b].href = save_path
                else:
                    to_download.append(
                        (local_items[i].assets[b].href, save_path))
                    missing_assets += 1
                    downloaded = False
                    del local_items[i].assets[b]
            except KeyError as e:
                pass
                # print(f'Asset {b} not found in item {local_items[i].id}')

        for b in ingnore_assets:
            try:
                del local_items[i].assets[b]
            except KeyError as e:
                pass

        if not downloaded:
            not_downloaded_items.append(local_items[i])

        # if set is only ignore_assets, then we don't have the data
        if set(local_items[i].assets.keys()) == set(ingnore_assets):
            idx_to_pop.append(i)

    for i in idx_to_pop[::-1]:
        local_items.pop(i)

    if report:
        if len(not_downloaded_items) == 0:  # should be a logger
            print('All data already downloaded.')
        else:
            print(f'{missing_assets} missing assets of {
                len(not_downloaded_items)} items to download.')

    return local_items, to_download


def check_sentinel2_data_exists_with_min_size(path):
    if not os_path.exists(path) or os_path.getsize(path)//1000000 < __get_filesize_mb_min(path):
        return False
    return True


def __get_filesize_mb_min(path):
    if path.endswith('10m.tif'):
        return 200
    elif path.endswith('20m.tif'):
        if path.endswith('SCL_20m.tif'):
            return 20
        else:
            return 40
    elif path.endswith('60m.tif'):
        return 5
    else:
        raise ValueError('Unknown resolution')


# def bulk_parallel_s2_bands(items, out_path, requested_bands):
#     tasks = []
#     hrefs = []
#     for item in items:
#         for band in requested_bands:
#             save_path = os_path.join(*[out_path,
#                                          item.assets[band].href.split("?")[0].split("/")[-6:]])
#             if not check_sentinel2_data_exists_with_min_size(save_path):
#                 hrefs.append((item.assets[band].href, save_path))

#     for h in hrefs:
#         tasks.append(get_asset(*h))
#     return tasks


@delayed
def get_asset(href, save_path):
    # faster but larger files
    makedirs(os_path.dirname(save_path), exist_ok=True)
    urlretrieve(pc_sign(href), save_path)
    # slower but smaller files, better rasterstats
    # (rioxarray.open_rasterio(pc_sign(href))
    #  .rio.to_raster(save_path))


# Define a style function for polygons
def style_function(feature):
    return {
        'fillColor': '#7FD0E1',  # fill for polygons
        'color': '#18334E',         # border for polygons
        'weight': 3,             # Line thickness
        'fillOpacity': 0.6,      # Opacity of the fill
    }

def style_function_points(feature):
    linear = cm.LinearColormap(["green", "yellow", "red"], vmin=0, vmax=21889)
    linear
    return {
        'fillColor': linear(feature['properties']['chunk_id']),  # fill for polygons
        'color': '#000000',         # border for polygons
        'weight': 0,             # Line thickness
        'fillOpacity': 0.6,      # Opacity of the fill
    }
# Define a highlight function for when the geometry is hovered over


def highlight_function(feature):
    return {
        'fillColor': '#ffaf00',  # Orange when highlighted
        'color': '#FBF97D',       # Yellow border when highlighted
        'weight': 5,             # Thicker border on hover
        'fillOpacity': 0.7,      # Slightly more opaque when hovered
    }


def leaflet_overview(items, chunktable=None):
    ids = []
    tile = []
    times = []
    assets = []
    geometry = []
    for i in items:
        ids.append(i.id)
        tile.append(i.properties['s2:mgrs_tile'])
        times.append(i.properties['datetime'])
        assets.append(list(i.assets.keys()))
        geometry.append(s_shape(i.geometry))
    gdf = gpd.GeoDataFrame({
        'id': ids,
        'tile': tile,
        'times': times,
        'assets': assets, },
        geometry=geometry,
        crs='epsg:4326'
    )
    # group by tiles and times
    gdf_tiles = gdf.groupby(['tile']).agg({
        # Merge geometries using unary_union
        'geometry': lambda x: s_shape(unary_union(x)),
    }).reset_index()
    gdf_tiles.set_geometry('geometry', inplace=True, crs='epsg:4326')
    gdf_ex = gdf.explode('assets')
    gdf_ex['assets'] = gdf_ex.assets.astype('category')

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        y = gdf.geometry.centroid.y.mean()
        x = gdf.geometry.centroid.x.mean()

    m = Map(location=[y, x], zoom_start=7, control_scale=True)
    LatLngPopup().add_to(m)

    if chunktable is not None:
        linear = cm.LinearColormap(["green", "yellow", "red"], vmin=0, vmax=21889)
        GeoJson(chunktable,
                name='MCChunks',
                show=True,
                control=False,
                marker=CircleMarker(
                   radius=4,
                   fill_color="orange",
                   fill_opacity=0.4,
                   color="black",
                   weight=1),
                style_function= lambda feature: {
                    'fillColor': linear(feature['properties']['chunk_id']),  # fill for polygons
                    'color': '#000000',         # border for polygons
                    'weight': 1,             # Line thickness
                    'fillOpacity': 0.6,      # Opacity of the fill
                },
                tooltip=GeoJsonTooltip(fields=["chunk_id", "lat_chunk", "lon_chunk"]),
                ).add_to(m)
    for t in np.unique(gdf.tile):
        popup = Popup()
        gdf_ex_sub = gdf_ex.loc[gdf_ex.tile == t]
        gdf_area = gdf_tiles.loc[gdf_tiles.tile == t]
        # make the chart
        tab = Chart(gdf_ex_sub).mark_point(filled=True).encode(
            x=alt_X('times:T', title='Time', axis=Axis(format="%Y %B")),
            y=alt_Y('assets', type='nominal', title='Assets'),
            color=alt_value('#18334E'),
        ).properties(
            width=600,
            title='Assets over time of tile {}'.format(t),
        ).interactive()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            pf_time = pd.DataFrame(
                {'time': gdf[gdf.tile == t].times.values[::-1],
                 'difference': (np.concatenate((np.diff(gdf[gdf.tile == t]
                                                        .times.values.astype('datetime64[D]')[::-1])
                                                        .astype(np.float64), [np.nan])))}
            )
        time_line = Chart(pf_time).mark_line().encode(
            x=alt_X('time:T', title='Time'),
            y=alt_Y('difference', title='Difference in days'),
            color=alt_value('orange'),
            strokeDash=alt_value([5, 5]),
            opacity=alt_value(0.7),
        ).properties(
            width=600, height=100,
        )
        vega_lite = VegaLite(
            tab + time_line,
            width="100%",
            height="100%",
        )
        vega_lite.add_to(popup)

        GeoJson(gdf_area,
                name='Tile',
                show=True,
                style_function=style_function,
                highlight_function=highlight_function,
                tooltip=GeoJsonTooltip(fields=['tile']),
                popup=popup,
                ).add_to(m)
    return m


def get_all_local_assets(out_path, collection='sentinel-2-l2a', requested_bands=None):
    files = listdir(out_path)
    files = [f[:27] + f[33:-5] for f in files if f.endswith('.SAFE')]

    stac = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = Client.open(stac)
    local_items = catalog.search(
        ids=files,
        collections=[collection],
    ).item_collection()
    all_local_avail_items, _ = check_request_against_local(
        local_items, out_path, requested_bands=requested_bands, report=False)
    return all_local_avail_items


def store_bands(band, cube, zarr_store, xy_index_slices, t_index_slices):
    cube[band].to_dataset(name=band).to_zarr(
        zarr_store, mode='a', write_empty_chunks=False,
        region={'time': slice(*t_index_slices),
                'latitude': slice(*xy_index_slices[:2]),
                'longitude': slice(*xy_index_slices[2:])})


@delayed
def store_chunks_to_zarr(dataset, zarr_store, b, t_index_slices, yx_index_slices, chunksize_xy, x, y):
    # TODO: handle last chunk in x and y, which may have less than chunksize_xy
    (dataset[b].isel(longitude=slice(x, x+chunksize_xy),
                     latitude=slice(y, y+chunksize_xy))
     .to_dataset(name=b)
     .to_zarr(zarr_store, mode='a',
              region={
                  'time': slice(*t_index_slices),
                  'latitude': slice(yx_index_slices[0]+y, yx_index_slices[0]+y+chunksize_xy),
                  'longitude': slice(yx_index_slices[2]+x, yx_index_slices[2]+x+chunksize_xy)}
              )
     )


def __lat_lon_from_geobox(geobox, x, y):
    a = geobox[x, y].affine
    return (a[2], a[5])


def chunk_table_from_geobox(geobox, chunksize_xy, aoi=None):
    num_lon_chunks = geobox.shape.y//chunksize_xy
    num_lat_chunks = geobox.shape.x//chunksize_xy

    lon_ch_id = list(range(chunksize_xy//2, num_lon_chunks *
                   chunksize_xy, chunksize_xy))
    lat_ch_id = list(range(chunksize_xy//2, num_lat_chunks *
                   chunksize_xy, chunksize_xy))

    lon_lat_chunks = list(product(lon_ch_id, lat_ch_id))
    lon_lat_coords = [__lat_lon_from_geobox(
        geobox, *i) for i in lon_lat_chunks]

    df_locs = pd.DataFrame({'lon_chunk': [l[0] for l in lon_lat_chunks],
                            'lat_chunk': [l[1] for l in lon_lat_chunks],
                            'lon_coord': [l[0] for l in lon_lat_coords],
                            'lat_coord': [l[1] for l in lon_lat_coords],
                            })

    gpd_locs = gpd.GeoDataFrame(df_locs,
                                geometry=gpd.points_from_xy(df_locs.lon_coord,
                                                            df_locs.lat_coord),
                                crs="EPSG:4326")
    if aoi:
        gpd_locs = gpd_locs.clip(aoi)
        gpd_locs = gpd_locs.sort_values(['lon_chunk', 'lat_chunk']).reset_index(drop=True)
        gpd_locs['chunk_id'] = gpd_locs.index
    return gpd_locs


# def get_chunk_table(dataset, aoi=None):
#     ilat = dataset.chunks['latitude']
#     ilon = dataset.chunks['longitude']
#     itime = dataset.chunks['time']

#     idlon = [sum(ilon[:i]) for i in range(len(ilon))]
#     idlat = [sum(ilat[:i]) for i in range(len(ilat))]
#     idtime = [sum(itime[:i]) for i in range(len(itime)+1)]

#     lon_mids = dataset.longitude.values[np.array(idlon[:-1]) + ilon[0]//2]
#     lat_mids = dataset.latitude.values[np.array(idlat[:-1]) + ilat[0]//2]
#     time_mids = dataset.time.values[np.array(idtime[:-1]) + itime[0]//2]

#     df_locs = pd.DataFrame(list(product(zip(time_mids, idtime),
#                                         zip(lat_mids, idlat),
#                                         zip(lon_mids, idlon))))

#     df_locs['time_mid'] = df_locs[0].apply(lambda x: x[0])
#     df_locs['time_chunk'] = df_locs[0].apply(lambda x: x[1])
#     df_locs['lat_mid'] = df_locs[1].apply(lambda x: x[0])
#     df_locs['lat_chunk'] = df_locs[1].apply(lambda x: x[1])
#     df_locs['lon_mid'] = df_locs[2].apply(lambda x: x[0])
#     df_locs['lon_chunk'] = df_locs[2].apply(lambda x: x[1])
#     df_locs = df_locs.drop(columns=[0, 1, 2])

#     gpd_locs = gpd.GeoDataFrame(df_locs,
#                                 geometry=gpd.points_from_xy(df_locs.lon_mid,
#                                                             df_locs.lat_mid),
#                                 crs="EPSG:4326")
#     if aoi:
#         gpd_locs = gpd_locs.clip(aoi)
#     return gpd_locs


# def get_chunk_index_around_latlon(lat, lon, chunktable):
#     return np.sqrt((chunktable['lat_mid'] - lat)**2 + (chunktable['lon_mid'] - lon)**2).argmin()


# def get_slices_at_index(index_, chunktable, dataset):

#     # t_max = chunktable['time_chunk'][index_] + dataset.chunks['time'][0]
#     # t_max = t_max if t_max < len(dataset.time) else len(dataset.time)
#     lat_max = chunktable['lat_chunk'][index_] + dataset.chunks['latitude'][0]
#     lat_max = lat_max if lat_max < len(
#         dataset.latitude) else len(dataset.latitude)
#     lon_max = chunktable['lon_chunk'][index_] + dataset.chunks['longitude'][0]
#     lon_max = lon_max if lon_max < len(
#         dataset.longitude) else len(dataset.longitude)

#     # time_slice = (chunktable['time_chunk'][index_], t_max)
#     lat_slice = (chunktable['lat_chunk'][index_], lat_max)
#     lon_slice = (chunktable['lon_chunk'][index_], lon_max)

#     return {  # 'time_slice': time_slice,
#         'lat_slice': lat_slice,
#         'lon_slice': lon_slice}


def get_slice_from_large_data(dataset, lat_slice, lon_slice, time_slice=None):
    if time_slice:
        return dataset.isel(time=slice(*time_slice),
                            latitude=slice(*lat_slice),
                            longitude=slice(*lon_slice))
    else:
        return dataset.isel(latitude=slice(*lat_slice),
                            longitude=slice(*lon_slice))


# def mc_from_chunk(lat, lon, dataset):
#     chunktable = get_chunk_table(dataset)
#     index_ = get_chunk_index_around_latlon(lat, lon, chunktable)
#     slices = get_slice_from_large_data(index_, chunktable, dataset)
#     return get_mc_from_large_data(slices, dataset)


# move tile data to folder structure
# for i in range(len(avail_items)):
#     for b in requested_bands:
#         #items[i].assets[b].href = os_path.join(*[out_path, folder_name] + items[i].assets[b].href.split("?")[0].split("/")[-6:])
#         assert items[i].assets[b].href.split("?")[0].split("/")[-1] == avail_items[i].assets[b].href.split("/")[-1]
#         new_loc = os_path.join(*[out_path, folder_name] + items[i].assets[b].href.split("?")[0].split("/")[-6:])

#         new_path = os_path.join(*[out_path, folder_name] + items[i].assets[b].href.split("?")[0].split("/")[-6:-1])


#
#         makedirs(new_path, exists_ok=True)

#         #move file to correct location
#         os_rename(avail_items[i].assets[b].href, new_loc)


# this runs, but is maybe not nice, 15 min for 20 gb
# from functools import partial
# import concurrent.futures

# fixed_function = partial(bulk_parallel_s2_bands, items=items, out_path=out_path,
#                          requested_bands=requested_bands, folder_name=folder_name)

# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = [executor.submit(fixed_function, x) for x in range(len(items))]
#     concurrent.futures.wait(futures)

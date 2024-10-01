import warnings

import pandas as pd
import geopandas as gpd
import numpy as np

from shapely.ops import unary_union
from shapely.geometry import shape as s_shape, Point, box as s_box

from folium import (Map, Popup, LatLngPopup, VegaLite, GeoJson,
                    GeoJsonTooltip, CircleMarker, LayerControl, FeatureGroup)
from altair import Chart, Axis, X as alt_X, Y as alt_Y, value as alt_value
import branca.colormap as cm

def leaflet_overview(items, chunktable=None, aoi=None):
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

    m = Map(control_scale=True)
    LatLngPopup().add_to(m)

    if aoi is not None:
        GeoJson(aoi,
                name='AOI',
                show=True,
                control=True,
                style_function=lambda feature: {
                    'color': '#000000',         # border for polygons
                    'weight': 2,             # Line thickness
                    'fillOpacity': 0.,      # Opacity of the fill
                },
                ).add_to(m)

    if chunktable is not None:
        linear = cm.LinearColormap(
            ['#fde725', '#b5de2b', '#6ece58', '#35b779', '#1f9e89',
             '#26828e', '#31688e', '#3e4989', '#482878', '#440154'],
            vmin=0, vmax=21889)
        GeoJson(chunktable,
                name='MCChunks',
                show=True,
                control=True,
                marker=CircleMarker(
                    radius=4,
                    fill_color="orange",
                    fill_opacity=0.4,
                    color="black",
                    weight=1),
                style_function=lambda feature: {
                    # fill for polygons
                    'fillColor': linear(feature['properties']['chunk_id']),
                    'color': '#000000',         # border for polygons
                    'weight': 1,             # Line thickness
                    'fillOpacity': 0.6,      # Opacity of the fill
                },
                tooltip=GeoJsonTooltip(
                    fields=["chunk_id", "lat_chunk", "lon_chunk"]),
                ).add_to(m)

    tile_list = FeatureGroup(name='Tiles', control=True)
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
                name=f'Tile {t}',
                show=True,
                control=True,
                style_function=lambda feature: {
                    'fillColor': '#7FD0E1',  # fill for polygons
                    'color': '#18334E',         # border for polygons
                    'weight': 3,             # Line thickness
                    'fillOpacity': 0.6,      # Opacity of the fill
                },
                highlight_function=lambda feature: {
                    'fillColor': '#ffaf00',  # Orange when highlighted
                    'color': '#FBF97D',       # Yellow border when highlighted
                    'weight': 5,             # Thicker border on hover
                    'fillOpacity': 0.7,      # Slightly more opaque when hovered
                },
                tooltip=GeoJsonTooltip(fields=['tile']),
                popup=popup,  # ENH: make the popup a funciton of the tile
                ).add_to(tile_list)
    tile_list.add_to(m)
    m.fit_bounds(m.get_bounds())
    LayerControl(collapsed=False).add_to(m)
    return m
import warnings

import pandas as pd
import geopandas as gpd
import numpy as np

from shapely import transform
from shapely.ops import unary_union
from shapely.geometry import shape as s_shape, Point, box as s_box, Polygon

from folium import (Map, Popup, LatLngPopup, VegaLite, GeoJson,
                    GeoJsonTooltip, CircleMarker, LayerControl, FeatureGroup,
                    GeoJsonPopup)
from altair import Chart, Axis, X as alt_X, Y as alt_Y, value as alt_value, data_transformers
import branca.colormap as cm



def leaflet_overview(gdf, chunktable=None, aoi=None, transform_to=None):
    data_transformers.disable_max_rows()
    # group by tiles and times
    gdf_tiles = gdf.groupby(['s2:mgrs_tile']).agg({
        # Merge geometries using unary_union
        'geometry': lambda x: s_shape(unary_union(x)),
    }).reset_index()

    gdf_tiles.set_geometry('geometry', inplace=True, crs=f'epsg:4326')
    gdf_ex = gdf[['s2:mgrs_tile', 'assets', 'datetime']].explode('assets')
    gdf_ex['assets'] = gdf_ex.assets.astype('category')

    m = Map(control_scale=True)
    LatLngPopup().add_to(m)

    if aoi is not None:
        if transform_to is not None:
            aoi = transform(aoi, transform_to, include_z=False)
            
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

    tile_list = FeatureGroup(name='Tiles', control=True)
    for t in np.unique(gdf_tiles['s2:mgrs_tile']):
        popup = Popup()
        gdf_ex_sub = gdf_ex.loc[gdf_ex['s2:mgrs_tile'] == t]
        # print(t, len(gdf_ex_sub))
        gdf_area = gdf_tiles.loc[gdf_tiles['s2:mgrs_tile'] == t]
        # make the chart

        # if t == '38PRQ':
        #     return gdf_ex_sub
        tab = Chart(gdf_ex_sub[['datetime', 'assets']]).mark_point(filled=True).encode(
            x=alt_X('datetime:T', title='Time', axis=Axis(format="%Y %B")),
            y=alt_Y('assets', type='nominal', title='Assets'),
            color=alt_value('#18334E'),
        ).properties(
            width=600,
            title='Assets over time of tile {}'.format(t),
        ).interactive()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            times = gdf[gdf['s2:mgrs_tile'] == t].datetime.values
            times.sort()
            pf_time = pd.DataFrame(
                {'time': times,
                 'difference':(np.concatenate((np.diff(times.astype('datetime64[D]').astype(np.int16)), [0])))}
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
            #tab.to_json(format='vega') + time_line.to_json(format='vega'),
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
                tooltip=GeoJsonTooltip(fields=['s2:mgrs_tile'], aliases=['Tile']),
                popup=popup,  # ENH: make the popup a funciton of the tile
                ).add_to(tile_list)
    tile_list.add_to(m)

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
                    fields=["chunk_id", "lat_coord", "lon_coord"]),
                popup=GeoJsonPopup(
                    fields=["chunk_id", "lat_coord", "lon_coord"]),
                ).add_to(m)
    
    m.fit_bounds(m.get_bounds())
    LayerControl(collapsed=False).add_to(m)
    return m
from functools import cached_property

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree
from shapely.geometry import Point, MultiPolygon


class SpatialGridCell:

    def __init__(self, index: int, major_tile: str, minor_tile: str, lonlat_geometry: MultiPolygon, grid: 'SpatialGrid'):
        self.index = index
        self.major_tile = major_tile
        self.minor_tile = minor_tile
        self.lonlat_geometry = lonlat_geometry
        self.grid = grid

    @property
    def tile(self):
        return f'{self.major_tile}_{self.minor_tile}'
    
    @property
    def center(self):
        return self.lonlat_geometry.centroid
    
    @property
    def bounds(self):
        return self.lonlat_geometry.bounds

    def __repr__(self):
        center_str = f'({self.center.y:.4f}°N, {self.center.x:.4f}°W)'
        return f'SpatialGridCell(index={self.index}, tile={self.tile}, center={center_str})'
    
    def __eq__(self, other):
        return self.index == other.index and self.grid == other.grid
    
    def __hash__(self):
        return hash((self.index, id(self.grid)))


class SpatialGrid:

    def __init__(self, name: str, file: str, min_lat=None, max_lat=None, min_lon=None, max_lon=None):
        self.name = name
        self.file = file
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon

    @cached_property
    def _parquet(self):
        gdf = gpd.read_parquet(self.file)
        gdf = gdf.astype({'epsg': np.int16})
        lon_slice = slice(self.min_lon, self.max_lon)
        lat_slice = slice(self.min_lat, self.max_lat)
        return gdf.cx[lon_slice, lat_slice]

    @cached_property
    def _exploded_parquet(self):
        return self._parquet.explode(index_parts=False)

    @cached_property
    def _tree(self):
        return STRtree(self._exploded_parquet.geometry)
    
    @cached_property
    def _major_tile_dict(self):
        dct = {}
        for idx, major_tile_id  in enumerate(self.major_tiles):
            dct.setdefault(major_tile_id, []).append(idx)
        return dct

    @cached_property
    def tiles(self) -> pd.CategoricalIndex:
        return pd.CategoricalIndex(self._parquet.tile)

    @cached_property
    def major_tiles(self) -> pd.CategoricalIndex:
        return self.tiles.map(lambda x: x.split('_', 1)[0])

    @cached_property
    def minor_tiles(self) -> pd.CategoricalIndex:
        return self.tiles.map(lambda x: x.split('_', 1)[1])

    def load(self):
        """
        Loads the spatial grid into memory and calculates any helper structures.
        """
        self._parquet
        self._exploded_parquet
        self._tree
        self._major_tile_dict

    def __len__(self):
        return len(self._parquet)

    def __getitem__(self, index) -> SpatialGridCell:
        if isinstance(index, int):
            row = self._parquet.iloc[index]
            major, minor = row.tile.split('_', 1)
            return SpatialGridCell(index, major, minor, lonlat_geometry=row.geometry, grid=self)
        elif isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        else:
            raise TypeError('Invalid argument type')

    def find_tile(self, tile: str) -> SpatialGridCell:
        idx = self.tiles.get_loc(tile)
        return self[idx]

    def find_major_tile(self, major_tile: str) -> list[SpatialGridCell]:
        indices = self._major_tile_dict.get(major_tile, None)
        return [self[i] for i in indices]

    def find_latlon(self, lat: float, lon: float) -> list[SpatialGridCell]:
        exploded_indices = self._tree.query(Point(lon, lat))
        indices = self._exploded_parquet.index[exploded_indices]
        coords = [self[i] for i in indices]
        return coords

    def __repr__(self):
        return f'SpatialGrid(name={self.name})'


UTM5KM = SpatialGrid('UTM5KM', '/Net/Groups/BGI/scratch/mzehner/code/S2_buckets_world_grid_5040.parquet', min_lat=-60, max_lat=75)


if __name__ == '__main__':
    UTM5KM.load()
    print(UTM5KM[:50])
    print(UTM5KM.find_major_tile('01UBS'))
    print(UTM5KM.find_major_tile('60WWV'))
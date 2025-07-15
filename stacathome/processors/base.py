import pystac
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry

from stacathome.providers import BaseProvider
from stacathome.stac import enclosing_geoboxes_per_grid
from stacathome.metadata import (get_static_metadata, has_static_metadata, 
                                 get_resampling_per_variable, get_variable_attributes)

_processor_registry: dict[tuple[str, str], "BaseProcessor"] = {}


class BaseProcessor:

    def filter_items(
        self, provider: BaseProvider, roi: Geometry, items: pystac.ItemCollection
    ) -> pystac.ItemCollection:
        """
        Filter items in the collection based on specific criteria.
        :param items: The item collection to filter.
        :return: Filtered item collection.
        """
        return items

    def load_items(self, provider: BaseProvider,
                   roi: Geometry,
                   items: pystac.ItemCollection,
                   variables: list[str] | None = None,
                   ) -> xr.Dataset:
        """
        Download items in the collection.
        :param provider: The provider to use for downloading.
        :param roi: The area of interest for the download.
        :param items: The item collection to download.
        :return: Item collection with downloaded items.
        """
        if not items:
            raise ValueError('No items provided')

        enclosing = enclosing_geoboxes_per_grid(items[0], roi)
        datasets = {}
        for group_nr, entry in enumerate(enclosing):
            asset_names = set(entry.assets) & set(variables) if variables else set(entry.assets)
            if not asset_names:
                continue
            
            # load the data for the given geobox and asset names
            datasets[str(group_nr)] = self.load_items_geoboxed(
                provider,
                geobox=entry.enclosing_box,
                items=items,
                variables=asset_names,
            )
        
        for group_nr in datasets.keys():
            datasets[group_nr] = datasets[group_nr].rename({'x': f'x_{group_nr}', 'y': f'y_{group_nr}'})

        cube = xr.merge(datasets.values())
        
        return cube

    def load_items_geoboxed(self, provider: BaseProvider,
                            geobox: GeoBox,
                            items: pystac.ItemCollection,
                            variables: list[str] | None = None,
                            resampling:dict[str, str] | None = None,
                            dtype: dict[str, float] | None = None,
                            ) -> xr.Dataset:
        """
        Download items in the collection.
        :param provider: The provider to use for downloading.
        :param geobox: The geobox defining the spatial extent and CRS of the output.
        :param items: The item collection to download.
        :return: Item collection with downloaded items.
        """
        if not items:
            raise ValueError('No items provided')

        metadata = None
        dtypes_static = None
        if has_static_metadata(provider.name, items[0].collection_id):
            metadata = get_static_metadata(provider.name, items[0].collection_id)
            dtypes_static = {v.name: v.dtype for v in metadata.variables.values()}
            
        attrs = get_variable_attributes(metadata, variables=variables)
        
        if not resampling:
            resampling = get_resampling_per_variable(metadata, geobox.resolution.x) if metadata \
                else {name: "nearest" for name in variables}
        if not dtype:
            # promote dtype if resampling is not 'nearest'
            if dtypes_static:
                dtype = {name: dtypes_static[name] if resampling[name] == 'nearest' else 'float32' for name in variables}
            
        for v in variables:
            attrs[v]['resampling'] = resampling[v]
            attrs[v]['dtype'] = dtype[v]
        
        loaded_xr = provider.load_items(items, geobox=geobox, variables=variables, resampling=resampling, dtype=dtype)
        
        for variable in loaded_xr.keys():
            loaded_xr[variable].attrs = attrs[variable]
        return loaded_xr

    def postprocess_data(self, provider: BaseProvider, roi: Geometry, data: xr.Dataset) -> xr.Dataset:
        """
        Post-process the downloaded data.
        :param data: The data to post-process.
        :return: Post-processed data.
        """
        return data


def register_default_processor(provider_name: str, collection: str, processor: BaseProcessor):
    key = (provider_name, collection)
    if key in _processor_registry:
        raise ValueError(f'Processor for {provider_name} and {collection} is already registered.')
    _processor_registry[(provider_name, collection)] = processor


def has_default_processor(provider_name: str, collection: str) -> bool:
    """
    Check if a default processor is registered for a given provider and collection.
    :param provider_name: The name of the provider.
    :param collection: The name of the collection.
    :return: True if a default processor is registered, False otherwise.
    """
    return (provider_name, collection) in _processor_registry


def get_default_processor(provider_name: str, collection: str) -> BaseProcessor | None:
    """
    Get the default processor for a given provider and collection.
    :param provider_name: The name of the provider.
    :param collection: The name of the collection.
    :return: The default processor for the specified provider and collection or None if not found.
    """
    key = (provider_name, collection)
    return _processor_registry.get(key)

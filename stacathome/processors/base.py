from typing import Callable

import pystac
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry

from stacathome.metadata import get_static_metadata
from stacathome.providers import BaseProvider
from stacathome.stac import enclosing_geoboxes_per_grid


ProcessorFactory = Callable[[], 'BaseProcessor']
_ProcessorRegistryKey = tuple[str, str]
_processor_factories: dict[_ProcessorRegistryKey, ProcessorFactory] = {}
_processor_instances: dict[_ProcessorRegistryKey, 'BaseProcessor'] = {}


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

    def load_items(
        self,
        provider: BaseProvider,
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

    def load_items_geoboxed(
        self,
        provider: BaseProvider,
        geobox: GeoBox,
        items: pystac.ItemCollection,
        variables: list[str] | None = None,
        resampling: dict[str, str] | None = None,
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
        variables = set(variables) if variables else None
        loaded_xr = provider.load_items(items, geobox=geobox, variables=variables, resampling=resampling, dtype=dtype)
        return loaded_xr

    def postprocess_data(self, provider: BaseProvider, roi: Geometry, data: xr.Dataset) -> xr.Dataset:
        """
        Post-process the downloaded data.
        :param data: The data to post-process.
        :return: Post-processed data.
        """
        return data


class SimpleProcessor(BaseProcessor):
    """
    Simple processor that looks up preferred resampling and dtype from static metadata, if available.
    Also adds metadata as attributes to the downloaded dataset.
    """

    def load_items_geoboxed(
        self,
        provider: BaseProvider,
        geobox: GeoBox,
        items: pystac.ItemCollection,
        variables: list[str] | None = None,
        resampling: dict[str, str] | None = None,
        dtype: dict[str, float] | None = None,
    ) -> xr.Dataset:
        if not items:
            raise ValueError('No items provided')

        collection = items[0].collection_id
        metadata = get_static_metadata(provider.name, collection)

        resampling = metadata.preferred_resampling_per_variable() if not resampling and metadata else {}
        dtype = metadata.dtype_per_variable() if not dtype and metadata else {}

        xr_dataset = super().load_items_geoboxed(
            provider=provider,
            geobox=geobox,
            items=items,
            variables=variables,
            resampling=resampling,
            dtype=dtype,
        )

        attrs_per_var = metadata.attributes_per_variable()
        for variable_name in xr_dataset.keys():
            var_attrs = attrs_per_var.get(variable_name, {})
            var_attrs['resampling'] = resampling.get('variable_name', 'nearest')
            xr_dataset[xr_dataset].attrs.update(var_attrs)

        return xr_dataset


def register_default_processor(provider_name: str, collection: str, factory: ProcessorFactory):
    if not callable(factory):
        raise TypeError('factory must be callable')

    key = (provider_name, collection)
    if key in _processor_factories:
        raise ValueError(f'Processor for {provider_name} and {collection} is already registered.')
    _processor_factories[(provider_name, collection)] = factory


def has_default_processor(provider_name: str, collection: str) -> bool:
    """
    Check if a default processor is registered for a given provider and collection.
    :param provider_name: The name of the provider.
    :param collection: The name of the collection.
    :return: True if a default processor is registered, False otherwise.
    """
    return (provider_name, collection) in _processor_factories


def get_default_processor(provider_name: str, collection: str) -> BaseProcessor | None:
    """
    Get the default processor for a given provider and collection.
    :param provider_name: The name of the provider.
    :param collection: The name of the collection.
    :return: The default processor for the specified provider and collection or None if not found.
    """
    key = (provider_name, collection)
    if key in _processor_instances:
        return _processor_instances[key]
    elif key in _processor_factories:
        processor = _processor_factories[key]()
        _processor_instances[key] = processor
        return processor
    else:
        return None

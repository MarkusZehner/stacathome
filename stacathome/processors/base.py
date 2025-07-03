import xarray as xr
import pystac
import shapely
from stacathome.providers import BaseProvider


_processor_registry: dict[tuple[str, str], "BaseProcessor"] = {}


class BaseProcessor:

    def filter_items(self, provider: BaseProvider, area_of_interest: shapely.Geometry, items: pystac.ItemCollection) -> pystac.ItemCollection:
        """
        Filter items in the collection based on specific criteria.
        :param items: The item collection to filter.
        :return: Filtered item collection.
        """
        return items


    def load_items(self, provider: BaseProvider, area_of_interest: shapely.Geometry, items: pystac.ItemCollection) -> xr.Dataset:
        """
        Download items in the collection.
        :param items: The item collection to download.
        :return: Item collection with downloaded items.
        """
        return provider.load(items)
        

    def postprocess_data(self, provider: BaseProvider, area_of_interest: shapely.Geometry, data: xr.Dataset) -> xr.Dataset:
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


def get_default_processor(provider_name: str, collection: str) -> BaseProcessor | None:
    """
    Get the default processor for a given provider and collection.
    :param provider_name: The name of the provider.
    :param collection: The name of the collection.
    :return: The default processor for the specified provider and collection or None if not found.
    """
    key = (provider_name, collection)
    return _processor_registry.get(key)

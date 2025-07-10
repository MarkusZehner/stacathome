from datetime import datetime
from typing import Callable

import pandas as pd
import pystac
import shapely
import xarray as xr
from odc.geo.geobox import GeoBox

from stacathome.metadata import CollectionMetadata


# Registry for provider classes and instances
_provider_classes: dict[str, Callable] = {}
_providers: dict[str, "BaseProvider"] = {}


class BaseProvider:
    """
    Represents a connection to a data provider. Repronsible  for session management and
    providing methods to request items, download granules, and download cubes.
    """

    def available_collections(self) -> list[str]:
        """
        Returns all collections available from this Provider.
        """
        raise NotImplementedError

    def get_metadata(self, collection: str) -> CollectionMetadata:
    def get_item(collection: str, item_id: str) -> pystac.Item | None:
        """
        Retrieves a STAC item from the specified collection.

        Args:
            collection (str): The name of the collection to retrieve the item from.
            item_id (str): The unique identifier of the item to retrieve.

        Returns:
            pystac.Item | None: The requested STAC item if found, otherwise None.

        Raises:
            KeyError: If the specified collection does not exist in the Provider.
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


    def _request_items(
        self,
        collection: str,
        starttime: datetime,
        endtime: datetime,
        area_of_interest: shapely.Geometry = None,
        limit: int = None,
        **kwargs,
    ) -> pystac.ItemCollection:
        """
        Requests items from the data provider based on the specified parameters.

        :param collection: The name of the collection to query.
        :param starttime: The start time for the query.
        :param endtime: The end time for the query.
        :param area_of_interest: The geographical area to query.
        :param limit: The maximum number of items to return.
        :param kwargs: Additional parameters for the request.
        :return: A collection of items matching the query.
        """
        raise NotImplementedError

    def request_items(
        self,
        collection: str,
        starttime: datetime | str,
        endtime: datetime | str,
        area_of_interest: shapely.Geometry = None,
        limit: int = None,
        **kwargs,
    ) -> pystac.ItemCollection:
        if not isinstance(collection, str):
            raise TypeError("Collection must be a string.")
        if isinstance(starttime, str):
            starttime = pd.to_datetime(starttime)
        if isinstance(endtime, str):
            endtime = pd.to_datetime(endtime)
        return self._request_items(
            collection=collection,
            starttime=starttime,
            endtime=endtime,
            area_of_interest=area_of_interest,
            limit=limit,
            **kwargs,
        )

    def load_items(self, items: pystac.ItemCollection, geobox: GeoBox | None = None, **kwargs) -> xr.Dataset:
        """
        Load items from the provider and returns them as a merged xr.Dataset.

        :param items: The item collection to load.
        :param geobox: Optional geobox to specify the spatial extent and crs of the output.
        :param kwargs: Additional parameters for loading.
        :return: Loaded item collection.
        """
        raise NotImplementedError

    def load_granule(self, item: pystac.Item, **kwargs) -> bytes:
        """
        Load a single granule from the provider into memory

        :param item: The item to load.
        :param kwargs: Additional parameters for loading.
        :return: Loaded granule as bytes.
        """
        raise NotImplementedError


def get_provider(provider_name: str) -> BaseProvider:
    provider = _providers.get(provider_name)

    if provider is None:
        provider_cls = _provider_classes.get(provider_name)
        if provider_cls is None:
            raise KeyError(f"Provider '{provider_name}' is not registered.")
        provider = provider_cls()
        if not isinstance(provider, BaseProvider):
            raise TypeError(f"Provider '{provider_name}' must be an instance of BaseProvider.")
        _providers[provider_name] = provider

    return provider


def register_provider(name, factory: Callable):
    """
    Registers a provider class with a given name.

    :param name: The name of the provider.
    :param factory: A callable that returns an instance of the provider.
    """
    if not callable(factory):
        raise ValueError("Factory must be a callable that returns an instance of BaseProvider.")
    _provider_classes[name] = factory

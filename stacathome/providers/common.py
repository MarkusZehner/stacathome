from datetime import datetime
from typing import Callable, Iterable

import pandas as pd
import pystac
import shapely
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.stac import load

from stacathome.metadata import CollectionMetadata


# Registry for provider classes and instances
ProviderFactory = Callable[[str], 'BaseProvider']
_provider_classes: dict[str, ProviderFactory] = {}
_providers: dict[str, "BaseProvider"] = {}


class BaseProvider:
    """
    Represents a connection to a data provider. Repronsible for session management and
    providing methods to request items, download granules, and download cubes.
    """

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def has_collection(self, collection: str) -> bool:
        """
        Check if the specified collection is supported by the provider.

        Args:
            collection (str): The name of the collection to check.

        Returns:
            bool: True if the collection is available, False otherwise.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        return collection in self.available_collections()

    def available_collections(self) -> list[str]:
        """
        List all collections available from this provider.

        Returns:
            list[str]: A list of collection names supported by the provider.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def get_metadata(self, collection: str) -> CollectionMetadata:
        """
        Retrieve metadata for a specified collection.

        Args:
            collection (str): The name of the collection for which metadata is requested.

        Returns:
            CollectionMetadata: The metadata associated with the specified collection.

        Raises:
            KeyError: If the specified collection does not exist.
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def get_item(self, collection: str, item_id: str) -> pystac.Item | None:
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
        roi: shapely.Geometry = None,
        limit: int = None,
        **kwargs,
    ) -> pystac.ItemCollection:
        raise NotImplementedError

    def request_items(
        self,
        collection: str,
        starttime: datetime | str,
        endtime: datetime | str,
        roi: shapely.Geometry = None,
        limit: int = None,
        **kwargs,
    ) -> pystac.ItemCollection:
        """
        Searches for items from the data provider matching the specified parameters.

        Args:
            collection (str): The name of the collection to query.
            starttime (datetime or str): The start time for the query. Can be a datetime object or an ISO8601 string.
            endtime (datetime or str): The end time for the query. Can be a datetime object or an ISO8601 string.
            roi (shapely.Geometry, optional): Region of interest. The geographical area to query. Defaults to None.
            limit (int, optional): The maximum number of items to return. Defaults to None.
            **kwargs: Additional parameters for the request.

        Returns:
            pystac.ItemCollection: A collection of items matching the query.
        """
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
            roi=roi,
            limit=limit,
            **kwargs,
        )

    def load_items(
        self,
        items: pystac.ItemCollection,
        geobox: GeoBox | None = None,
        variables: Iterable[str] | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Loads items from the provider and returns them as a merged xarray.Dataset.

        Args:
            items (pystac.ItemCollection): The collection of STAC items to load.
            geobox (GeoBox, optional): Specifies the spatial extent and coordinate reference system (CRS) of the output dataset. Defaults to None.
            variables (Iterable[str], optional): Specifies which variables to load from the items.
            **kwargs: Additional keyword arguments for loading.

        Returns:
            xr.Dataset: The loaded and merged dataset from the provided items.
        """
        raise NotImplementedError

    def load_granule(
        self,
        item: pystac.Item,
        variables: list[str] | None = None,
        out_dir: str | None = None,
        threads: int = 1,
        **kwargs,
    ) -> bytes:
        """
        Loads granules of item variables from the provider to disk.

        Args:
            item (pystac.Item): The item representing the granule to load.
            **kwargs: Additional keyword arguments for loading.

        Returns:
            bytes: The loaded granule as a bytes object.
        """
        raise NotImplementedError


class SimpleProvider(BaseProvider):
    """
    Simple provider to collect common functions.
    """

    def __init__(self, name):
        super().__init__(name)
        self.sign = None

    def load_items(
        self,
        items: pystac.ItemCollection,
        geobox: GeoBox | None = None,
        variables: Iterable[str] | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Loads items from the provider and returns them as a merged xarray.Dataset.

        Args:
            items (pystac.ItemCollection): The collection of STAC items to load.
            geobox (GeoBox, optional): Specifies the spatial extent and coordinate reference system (CRS) of the output dataset. Defaults to None.
            variables (Iterable[str], optional): Specifies which variables to load from the items.
            **kwargs: Additional keyword arguments for loading.

        Returns:
            xr.Dataset: The loaded and merged dataset from the provided items.
        """
        if not items:
            raise ValueError('No items provided for loading.')

        variables = set(variables) if variables else None
        groupby = kwargs.pop('groupby', 'id')

        data = load(
            items=items,
            bands=variables,
            geobox=geobox,
            groupby=groupby,
            # This is important for the filtering to be used!
            # By default items are sorted by time, id within each group to make pixel fusing order deterministic.
            # Setting this flag to True will instead keep items within each group in the same order as supplied,
            # so that one can implement arbitrary priority for pixel overlap cases.
            preserve_original_order=True,
            patch_url=self.sign,
            **kwargs,
        )
        # sort data by time
        data = data.sortby('time')
        return data


def get_provider(provider_name: str) -> BaseProvider:
    """
    Retrieve a provider instance by name.

    If the provider instance is already registered in the internal cache, it is returned.
    Otherwise, the provider class is looked up, instantiated and then returned.

    Args:
        provider_name (str): The name of the provider to retrieve.

    Returns:
        BaseProvider: An instance of the requested provider.

    Raises:
        KeyError: If the provider name is not registered.
    """
    provider = _providers.get(provider_name)

    if provider is None:
        provider_cls = _provider_classes.get(provider_name)
        if provider_cls is None:
            raise KeyError(f"Provider '{provider_name}' is not registered")
        provider = provider_cls(provider_name)
        if not isinstance(provider, BaseProvider):
            raise TypeError(f"Provider '{provider_name}' must be an instance of BaseProvider")
        if provider.name != provider_name:
            raise ValueError(
                f"Provider was registered under name '{provider_name}' but returns '{provider.name} as name attribute'"
            )
        _providers[provider_name] = provider

    return provider


def register_provider(provider_name: str, factory: ProviderFactory):
    """
    Registers a provider class with a given name.

    Args:
        provider_name (str): The name of the provider.
        factory (ProviderFactory): A callable that returns an instance of the provider. Must accept provider_name as first argument.

    Raises:
        ValueError: If the factory is not callable or the provider name is already registered.
    """
    if not callable(factory):
        raise TypeError('factory must be callable')
    if provider_name in _provider_classes:
        raise ValueError(f"Provider '{provider_name}' is already registered.")
    _provider_classes[provider_name] = factory

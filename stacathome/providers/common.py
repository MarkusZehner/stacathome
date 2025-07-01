from datetime import datetime
from typing import Callable

import pystac


_provider_classes: dict[str, Callable] = {}

_providers: dict[str, "BaseProvider"] = {}


def get_provider(provider_name: str) -> "BaseProvider":
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


class BaseProvider:
    """
    Represents a connection to a data provider. Repronsible  for session management and
    providing methods to request items, download granules, and download cubes.
    """

    def connect(self):
        """
        Establishes a connection to the data provider.
        This method should be overridden by subclasses to implement specific connection logic.
        """
        raise NotImplementedError

    def close(self):
        """
        Closes any open connections or sessions.
        """
        raise NotImplementedError

    def request_items(
        self, collection: str, starttime: datetime, location: any = None, max_retry: int = 5, **kwargs
    ) -> pystac.ItemCollection:
        raise NotImplementedError

    def download_granules_to_file(self, href_path_tuples: list[tuple]):  #
        raise NotImplementedError

    def download_cube(self, parameters):
        raise NotImplementedError

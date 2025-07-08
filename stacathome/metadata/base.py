import pprint
from dataclasses import field
from typing import Iterable, Optional

from pydantic.dataclasses import dataclass as pydantic_dataclass


_metadata_registry: dict[tuple[str, str], 'CollectionMetadata'] = {}


@pydantic_dataclass(frozen=True)
class Variable:
    name: str
    longname: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    roles: list[str] = field(default_factory=list)

    dtype: Optional[str] = None
    preferred_resampling: Optional[str] = None  # possible values: nearest, bilinear, mode
    nodata_value: Optional[int | float] = None
    offset: Optional[int | float] = None
    scale: Optional[int | float] = None
    spatial_resolution: Optional[float] = None
    center_wavelength: Optional[float] = None
    full_width_half_max: Optional[float] = None


class CollectionMetadata:

    def __init__(self, *variables: Iterable[Variable]):
        self._variables = {var.name: var for var in variables}

    @property
    def variables(self):
        return self._variables

    def available_variables(self) -> list[str]:
        return list(self.variables)

    def has_variable(self, variable: str) -> bool:
        return variable in self.variables

    def get_variable(self, variable: str) -> Variable | None:
        return self.variables.get(variable)

    def aspystr(self):
        """
        Returns this object as pretty-formated and valid python string.
        This method differs from __repr__ with regards to formattin but is functional equivalent.
        """
        pp_vars = []
        for var in self.variables.values():
            pp_vars.append(pprint.pformat(var, compact=True, sort_dicts=False, width=120))
        var_str = ',\n'.join(pp_vars)
        str = f'{self.__class__.__name__}({var_str})'
        return str.replace('=nan', "=float('nan')")  # workaround for https://bugs.python.org/issue1732212

    def __str__(self):
        return self.aspystr()

    def __repr__(self):
        var_str = ','.join(repr(var) for var in self.variables.values())
        str = f'{self.__class__.__name__}({var_str})'
        return str.replace('=nan', "=float('nan')")  # workaround for https://bugs.python.org/issue1732212


def register_static_metadata(provider_name: str, collection: str, metadata: CollectionMetadata):
    key = (provider_name, collection)
    if key in _metadata_registry:
        raise ValueError(f'Processor for {provider_name} and {collection} is already registered.')
    _metadata_registry[(provider_name, collection)] = metadata


def has_static_metadata(provider_name: str, collection: str) -> bool:
    return (provider_name, collection) in _metadata_registry


def get_static_metadata(provider_name: str, collection: str) -> CollectionMetadata | None:
    key = (provider_name, collection)
    return _metadata_registry.get(key)

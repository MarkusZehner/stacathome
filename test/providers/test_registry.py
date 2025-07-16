import pytest
from stacathome.providers import BaseProvider, get_provider, register_provider


class FakeProvider(BaseProvider):
    pass


class TestProviderRegistry:

    def test_register_provider(self):
        with pytest.raises(KeyError):
            get_provider('test_provider')

        with pytest.raises(TypeError):
            register_provider('test_provider', 1212314)
        with pytest.raises(TypeError):
            register_provider('test_provider', 'FakeProvider')

        register_provider('test_provider', FakeProvider)

        provider = get_provider('test_provider')
        assert isinstance(provider, FakeProvider)
        assert provider.name == 'test_provider'

        provider2 = get_provider('test_provider')
        assert provider is provider2  # ensure the same instance is returned

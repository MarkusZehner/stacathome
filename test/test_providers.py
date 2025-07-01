import pytest

from stacathome.providers import get_provider, register_provider, BaseProvider


class FakeProvider(BaseProvider):
    pass


class TestProviderRegistry:

    def test_register_provider(self):
        with pytest.raises(KeyError):
            get_provider('test_provider')

        register_provider('test_provider', FakeProvider)

        provider = get_provider('test_provider')
        assert isinstance(provider, FakeProvider)

        provider2 = get_provider('test_provider')
        assert provider is provider2  # Ensure the same instance is returned
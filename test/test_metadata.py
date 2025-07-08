import pytest
from stacathome.metadata import (
    Variable,
    CollectionMetadata,
    register_static_metadata,
    has_static_metadata,
    get_static_metadata,
)

class TestVariable:

    def test_immutable(self):
        var = Variable(name='B01', longname='Band 1')
        with pytest.raises(AttributeError):
            var.name = 'B02'

    def test_input_validation(self):
        # name is required
        with pytest.raises(ValueError):
            Variable()
    
        # dtype can be any string or None
        var = Variable(name='B01', dtype='uint16')
        assert var.dtype == 'uint16'

        var = Variable(name='B01', dtype=None)
        assert var.dtype is None 

        with pytest.raises(ValueError):
            Variable(name='B01', dtype=12345)

        # roles defaults to empty list
        var = Variable(name='B01')
        assert var.roles == []

        with pytest.raises(ValueError):
            Variable(name='B01', roles=12345)


class TestCollectionMetadata:

    def test_variable_storage_and_access(self):
        v1 = Variable(name='B01')
        v2 = Variable(name='B02')
        meta = CollectionMetadata(v1, v2)
        assert meta.has_variable('B01')
        assert meta.get_variable('B02') == v2
        assert set(meta.available_variables()) == {'B01', 'B02'}

    def test_aspystr_and_repr(self):
        v = Variable(name='B01')
        meta = CollectionMetadata(v)
        s = meta.aspystr()
        assert 'CollectionMetadata' in s
        assert 'B01' in s
        r = repr(meta)
        assert 'CollectionMetadata' in r
        assert 'B01' in r


class TestMetadataRegistry:

    def setup_method(self):
        # Clear the registry before each test
        # Access the private registry for test isolation
        from stacathome.metadata.base import _metadata_registry
        _metadata_registry.clear()

    def test_register_and_has_static_metadata(self):
        var = Variable(name='B01', longname='Band 1')
        meta = CollectionMetadata(var)
        assert not has_static_metadata('provider1', 'collection1')
        register_static_metadata('provider1', 'collection1', meta)
        assert has_static_metadata('provider1', 'collection1')

    def test_register_duplicate_raises(self):
        var = Variable(name='B01')
        meta = CollectionMetadata(var)
        register_static_metadata('provider2', 'collection2', meta)
        with pytest.raises(ValueError):
            register_static_metadata('provider2', 'collection2', meta)

    def test_get_static_metadata_returns_correct_type_and_value(self):
        var = Variable(name='B02', description='desc')
        meta = CollectionMetadata(var)
        register_static_metadata('provider3', 'collection3', meta)
        result = get_static_metadata('provider3', 'collection3')
        assert isinstance(result, CollectionMetadata)
        assert result.get_variable('B02').description == 'desc'

    def test_get_static_metadata_returns_none_for_missing(self):
        result = get_static_metadata('providerX', 'collectionX')
        assert result is None
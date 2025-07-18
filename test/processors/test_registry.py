import pytest
from stacathome.processors import (
    BaseProcessor,
    get_default_processor,
    has_default_processor,
    register_default_processor,
)


class FakeProcessor(BaseProcessor):
    pass


class TestProcessorRegistry:

    def test_register_provider(self):
        assert get_default_processor('test_provider', 'test_collection') is None
        assert not has_default_processor('test_provider', 'test_collection')

        # Only callables can be registers
        with pytest.raises(TypeError):
            register_default_processor('test_provider', 'test_collection', 1224124)
        with pytest.raises(TypeError):
            register_default_processor('test_provider', 'test_collection', 'FakeProcessor')

        register_default_processor('test_provider', 'test_collection', FakeProcessor)
        assert has_default_processor('test_provider', 'test_collection')

        processor1 = get_default_processor('test_provider', 'test_collection')
        assert isinstance(processor1, FakeProcessor)

        processor2 = get_default_processor('test_provider', 'test_collection')
        assert processor1 is processor2  # Ensure the same instance is returned

# import pytest
# from pathlib import Path


# from stacathome.auth.handler import SecretStore

# class TestSecretStore:
#     name = 'test_secrets.env'
#     store = SecretStore(custom_name=name)

#     assert (Path.home() / name).exists() is False

#     store.create()
    
#     assert (Path.home() / name).exists()

#     store.add_key('test_name', 'test_key', 'test_value')

#     assert store.get_key('test_name') == ('test_key', 'test_value')

#     store.delete_key('test_name', 'test_key')

#     with pytest.raises(KeyError):
#         store.get_key("test_name")

import pytest
import tempfile
from pathlib import Path
from stacathome.auth.handler import SecretStore  


@pytest.fixture
def temp_env_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".stacathome_secrets.env"
        yield path


def test_init_existing_file(temp_env_file):
    temp_env_file.write_text("")
    store = SecretStore(custom_name=temp_env_file, mock=True)
    assert store.filename == temp_env_file
    assert store.exists() is True

def test_init_missing_file(temp_env_file):
    store = SecretStore(custom_name=temp_env_file, mock=True)
    assert store.filename.exists() is False

def test_create_creates_file(tmp_path):
    home = tmp_path
    expected = home / ".stacathome_secrets.env"
    store = SecretStore()
    store.filename = expected  # manually set filename to controlled location
    store.create()
    assert expected.exists()
    assert expected.stat().st_mode & 0o777 == 0o600


def test_add_key_and_get_key(temp_env_file):
    temp_env_file.touch()
    store = SecretStore(custom_name=temp_env_file, mock=True)
    store.add_key("user1", "apikey", "secret123")
    key, val = store.get_key("user1")
    assert key == "apikey"
    assert val == "secret123"


def test_add_key_duplicate_without_overwrite(temp_env_file):
    temp_env_file.touch()
    store = SecretStore(custom_name=temp_env_file, mock=True)
    store.add_key("user1", "apikey", "secret123")
    with pytest.raises(KeyError):
        store.add_key("user1", "apikey", "another_secret")


def test_add_key_duplicate_with_overwrite(temp_env_file):
    temp_env_file.touch()
    store = SecretStore(custom_name=temp_env_file, mock=True)
    store.add_key("user1", "apikey", "secret123")
    store.add_key("user1", "apikey", "another_secret", overwrite=True)
    key, val = store.get_key("user1")
    assert val == "another_secret"


def test_get_key_missing(temp_env_file):
    temp_env_file.touch()
    store = SecretStore(custom_name=temp_env_file, mock=True)
    with pytest.raises(KeyError):
        store.get_key("no_user")


def test_get_key_multiple_keys(temp_env_file):
    temp_env_file.write_text("user1:key1=val1\nuser1:key2=val2\n")
    store = SecretStore(custom_name=temp_env_file, mock=True)
    with pytest.raises(ValueError):
        store.get_key("user1")


def test_delete_key(temp_env_file):
    temp_env_file.write_text("user1:key1=val1\n")
    store = SecretStore(custom_name=temp_env_file, mock=True)
    store.delete_key("user1", "key1")
    assert store._read_all() == {}


def test_delete_key_not_found(temp_env_file):
    temp_env_file.write_text("user1:key1=val1\n")
    store = SecretStore(custom_name=temp_env_file, mock=True)
    with pytest.raises(KeyError):
        store.delete_key("user1", "missing_key")


def test_delete_key_name_not_found(temp_env_file):
    temp_env_file.write_text("user1:key1=val1\n")
    store = SecretStore(custom_name=temp_env_file, mock=True)
    with pytest.raises(KeyError):
        store.delete_key("user2", "key1")

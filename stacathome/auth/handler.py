from pathlib import Path


class SecretStore:
    def __init__(self, custom_name: str | None = None, mock=False):
        name = custom_name if custom_name else '.stacathome_secrets.env'
        self.filename = Path.home() / name

        if mock:
            self.filename = name
        # if not self.filename.exists():
        #     self.filename = None

    def exists(self):
        return self.filename.exists()

    def create(self):
        # self.filename = Path.home() / '.stacathome_secrets.env'
        if not self.filename.exists():
            self.filename.touch(mode=0o600)

    def _read_all(self):
        secrets = {}
        with self.filename.open('r') as f:
            for line in f:
                line = line.strip()
                if not line or '=' not in line or ':' not in line:
                    continue
                name_key, value = line.split('=', 1)
                name, key = name_key.split(':', 1)
                if name not in secrets:
                    secrets[name] = {}
                secrets[name][key] = value
        return secrets

    def _write_all(self, secrets):
        with self.filename.open('w') as f:
            for name, keys in secrets.items():
                for key, value in keys.items():
                    f.write(f"{name}:{key}={value}\n")

    def add_key(self, name, key, value, overwrite=False):
        secrets = self._read_all()
        if name in secrets and not overwrite:
            raise KeyError(f"A key already exists under name '{name}', set overwrite=True to replace")
        secrets.setdefault(name, {})[key] = value
        self._write_all(secrets)

    def get_key(self, name):
        secrets = self._read_all()
        if name not in secrets:
            raise KeyError(f"No keys found under name '{name}'.")
        keys = secrets[name]
        if len(keys) != 1:
            raise ValueError(f"Expected exactly one key under name '{name}', found {len(keys)}.")
        key, value = next(iter(keys.items()))
        return (key, value)

    def delete_key(self, name, key):
        secrets = self._read_all()
        if name not in secrets or key not in secrets[name]:
            raise KeyError(f"Key '{key}' under name '{name}' not found.")

        del secrets[name][key]

        # Remove name entry if it has no more keys
        if not secrets[name]:
            del secrets[name]

        self._write_all(secrets)

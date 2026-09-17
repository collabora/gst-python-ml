from pathlib import Path

G2G_ONLY_FILE = "test_g2g_backend.py"


# one process hosts one backend and that file picks g2g at import
def pytest_ignore_collect(collection_path, config):
    if collection_path.name != G2G_ONLY_FILE:
        return None
    return not any(
        Path(arg.split("::")[0]).name == G2G_ONLY_FILE for arg in config.args
    )

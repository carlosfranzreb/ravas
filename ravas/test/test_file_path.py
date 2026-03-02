import os
from stream_processing.utils import get_config_path, resolve_file_path, APPLICATION_DIR

"""
Test the path utility functions `resolve_file_path` and `get_config_path`.

This test verifies that:
- `resolve_file_path` returns absolute paths unchanged.
- `resolve_file_path` correctly resolves relative paths against `APPLICATION_DIR`.
- `get_config_path` returns the full path to a config file if it exists
  in the `configs/` directory.
- `get_config_path` returns the provided filename with a `.yaml`
  extension if the config file does not exist.
"""


def test_resolve_file_path_absolute():
    """
    Absolute paths should be returned unchanged.
    """
    abs_path = "/tmp/somefile.txt"
    resolved = resolve_file_path(abs_path)
    assert resolved == abs_path


def test_resolve_file_path_relative():
    """
    Relative paths should be resolved against the APPLICATION_DIR.
    """
    rel_path = "subdir/file.txt"
    expected = os.path.realpath(os.path.join(APPLICATION_DIR, rel_path))
    resolved = resolve_file_path(rel_path)
    assert resolved == expected


def test_get_config_path_existing_file(monkeypatch, tmp_path):
    """
    If a config file exists in the default configs/ directory,
    get_config_path should return the full path to that file.
    """
    # Create configs directory inside temporary path
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Create a test config file
    filename = "testconfig.yaml"
    filepath = configs_dir / filename
    filepath.write_text("dummy: 1")

    # Patch APPLICATION_DIR inside the utils module
    import stream_processing.utils as utils

    monkeypatch.setattr(utils, "APPLICATION_DIR", str(tmp_path))

    result = get_config_path("testconfig")

    assert result == str(filepath)


def test_get_config_path_nonexistent_file():
    """
    If the file does not exist in configs/, return the relative file name with .yaml extension.
    """
    filename = "nonexistent_config"
    result = get_config_path(filename)
    assert result == filename + ".yaml"

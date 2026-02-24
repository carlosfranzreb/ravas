import unittest
import os
import tempfile
from stream_processing.utils import get_config_path, resolve_file_path, APPLICATION_DIR


class TestPathUtils(unittest.TestCase):

    def test_resolve_file_path_absolute(self):
        """
        Absolute paths should be returned unchanged.
        """
        abs_path = "/tmp/somefile.txt"
        resolved = resolve_file_path(abs_path)
        self.assertEqual(resolved, abs_path)

    def test_resolve_file_path_relative(self):
        """
        Relative paths should be resolved against the APPLICATION_DIR.
        """
        rel_path = "subdir/file.txt"
        expected = os.path.realpath(os.path.join(APPLICATION_DIR, rel_path))
        resolved = resolve_file_path(rel_path)
        self.assertEqual(resolved, expected)

    def test_get_config_path_existing_file(self):
        """
        If a config file exists in the default configs/ directory,
        get_config_path should return the full path to that file.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            configs_dir = os.path.join(tmpdir, "configs")
            os.makedirs(configs_dir, exist_ok=True)

            # create a test config file
            filename = "testconfig.yaml"
            filepath = os.path.join(configs_dir, filename)
            with open(filepath, "w") as f:
                f.write("dummy: 1")

            # temporarily patch APPLICATION_DIR to tmpdir for testing
            original_app_dir = os.environ.get("APPLICATION_DIR")
            try:
                # patch the APPLICATION_DIR in the utils module
                from ravas.stream_processing import utils
                utils.APPLICATION_DIR = tmpdir
                result = get_config_path("testconfig")
                self.assertEqual(result, filepath)
            finally:
                # restore original APPLICATION_DIR
                utils.APPLICATION_DIR = APPLICATION_DIR if original_app_dir is None else original_app_dir

    def test_get_config_path_nonexistent_file(self):
        """
        If the file does not exist in configs/, return the relative file name with .yaml extension.
        """
        filename = "nonexistent_config"
        result = get_config_path(filename)
        self.assertEqual(result, filename + ".yaml")


if __name__ == "__main__":
    unittest.main()

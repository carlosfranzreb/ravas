import os
import shutil
import unittest
import torch

# Ensure your imports match your project structure
from stream_processing.utils import compute_target_features


class TestComputeTargetFeatures(unittest.TestCase):
    """
    Test the standalone compute_target_features function.
    """

    def setUp(self):
        """Setup the temp folder for the target feature and cluster checkpoints."""
        self.test_dir = os.path.join(os.path.dirname(__file__), "temp_target_feats")
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a dummy features file used across multiple tests
        self.dummy_feats = torch.rand(100, 128)
        self.temp_feats_path = os.path.join(self.test_dir, "features.pt")
        torch.save(self.dummy_feats, self.temp_feats_path)

    def tearDown(self):
        """Remove the temp feature folder."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_cluster_creation(self):
        """
        Test that clusters are correctly computed, returned, and saved to disk.
        """
        n_cluster = 10
        # Call the standalone function
        cluster_feats = compute_target_features(
            target_features_path=self.temp_feats_path,
            n_cluster=n_cluster,
            use_context=True,
        )

        # Verify output shape
        self.assertEqual(cluster_feats.shape, (n_cluster, 128))

        # Verify file creation
        # Path logic: dirname/features_cluster/10.pt
        expected_path = os.path.join(
            self.test_dir, "features_cluster", f"{n_cluster}.pt"
        )
        self.assertTrue(
            os.path.exists(expected_path), f"Cluster file not found at {expected_path}"
        )

    def test_cluster_reload(self):
        """
        Test that the function loads an existing cluster file instead of recomputing.
        """
        n_cluster = 5

        # 1. First call to create the file
        first_run = compute_target_features(self.temp_feats_path, n_cluster, True)

        # 2. Second call should load the file
        # We can verify this by checking if the results are identical
        second_run = compute_target_features(self.temp_feats_path, n_cluster, True)

        self.assertTrue(torch.allclose(first_run, second_run))

    def test_no_clustering_conditions(self):
        """
        Test cases where clustering should be skipped:
        1. n_cluster == 0
        2. use_context == False
        """
        # Case 1: n_cluster is 0
        feats_0 = compute_target_features(
            self.temp_feats_path, n_cluster=0, use_context=True
        )
        self.assertTrue(torch.allclose(self.dummy_feats, feats_0))

        # Case 2: use_context is False
        feats_no_context = compute_target_features(
            self.temp_feats_path, n_cluster=10, use_context=False
        )
        self.assertTrue(torch.allclose(self.dummy_feats, feats_no_context))

        # Ensure no cluster folder was accidentally created for the n_cluster=0 case
        cluster_file_0 = os.path.join(self.test_dir, "features_cluster", "0.pt")
        self.assertFalse(os.path.exists(cluster_file_0))

    def test_directory_creation(self):
        """Verify that the nested cluster directory is created automatically."""
        cluster_dir = os.path.join(self.test_dir, "features_cluster")

        # Before calling, it shouldn't exist
        self.assertFalse(os.path.exists(cluster_dir))

        compute_target_features(self.temp_feats_path, n_cluster=2, use_context=True)

        # After calling, it should exist
        self.assertTrue(os.path.isdir(cluster_dir))


if __name__ == "__main__":
    unittest.main()

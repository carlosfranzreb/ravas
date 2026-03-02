import torch
import pytest

from stream_processing.utils import compute_target_features

"""
Test clustering of compute_target_features
This test covers the following scenarios with temporary directory and dummy features:
1. Cluster Creation: Verify that clusters are correctly computed, returned, and saved to disk.
2. Cluster Reload: Ensure that if a cluster file already exists, it is loaded instead of recomputed.
3. No Clustering Conditions: Test cases where clustering should be skipped (n_cluster=0) or use_context=False).
4. Directory Creation: Confirm that the necessary directories are created if they don't exist.
"""


@pytest.fixture
def temp_feature_dir(tmp_path):
    """
    Create a temporary directory with a dummy features file.
    Automatically cleaned up by pytest.
    """
    test_dir = tmp_path / "temp_target_feats"
    test_dir.mkdir(parents=True, exist_ok=True)

    dummy_feats = torch.rand(100, 128)
    temp_feats_path = test_dir / "features.pt"
    torch.save(dummy_feats, temp_feats_path)

    return {
        "test_dir": test_dir,
        "dummy_feats": dummy_feats,
        "temp_feats_path": temp_feats_path,
    }


def test_cluster_creation(temp_feature_dir):
    """
    Test that clusters are correctly computed, returned, and saved to disk.
    """
    test_dir = temp_feature_dir["test_dir"]
    temp_feats_path = temp_feature_dir["temp_feats_path"]

    n_cluster = 10

    cluster_feats = compute_target_features(
        target_features_path=str(temp_feats_path),
        n_cluster=n_cluster,
        use_context=True,
    )

    # Verify output shape
    assert cluster_feats.shape == (n_cluster, 128)

    # Verify file creation
    expected_path = test_dir / "features_cluster" / f"{n_cluster}.pt"
    assert expected_path.exists(), f"Cluster file not found at {expected_path}"


def test_cluster_reload(temp_feature_dir):
    """
    Test that the function loads an existing cluster file instead of recomputing.
    """
    temp_feats_path = temp_feature_dir["temp_feats_path"]
    n_cluster = 5

    # First call creates the file
    first_run = compute_target_features(str(temp_feats_path), n_cluster, True)

    # Second call should load the same file
    second_run = compute_target_features(str(temp_feats_path), n_cluster, True)

    assert torch.allclose(first_run, second_run)


def test_no_clustering_conditions(temp_feature_dir):
    """
    Test cases where clustering should be skipped:
    1. n_cluster == 0
    2. use_context == False
    """
    test_dir = temp_feature_dir["test_dir"]
    dummy_feats = temp_feature_dir["dummy_feats"]
    temp_feats_path = temp_feature_dir["temp_feats_path"]

    # Case 1: n_cluster == 0
    feats_0 = compute_target_features(
        str(temp_feats_path), n_cluster=0, use_context=True
    )
    assert torch.allclose(dummy_feats, feats_0)

    # Case 2: use_context is False
    feats_no_context = compute_target_features(
        str(temp_feats_path), n_cluster=10, use_context=False
    )
    assert torch.allclose(dummy_feats, feats_no_context)

    # Ensure no cluster file was created for n_cluster=0
    cluster_file_0 = test_dir / "features_cluster" / "0.pt"
    assert not cluster_file_0.exists()


def test_directory_creation(temp_feature_dir):
    """Verify that the nested cluster directory is created automatically."""
    test_dir = temp_feature_dir["test_dir"]
    temp_feats_path = temp_feature_dir["temp_feats_path"]

    cluster_dir = test_dir / "features_cluster"

    # Before calling, it shouldn't exist
    assert not cluster_dir.exists()

    compute_target_features(str(temp_feats_path), n_cluster=2, use_context=True)

    # After calling, it should exist
    assert cluster_dir.is_dir()

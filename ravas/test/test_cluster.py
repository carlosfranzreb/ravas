import os
import shutil
import unittest
import torch

from ..stream_processing.utils import TargetFeats

class TestTargetFeatures(unittest.TestCase):
    """
    Test the get_cluster function of the TargetFeats class.

    This test suite ensures that:
    - Features can be loaded from a path and clustered correctly.
    - Clusters are saved and reloaded properly.
    - Original features are returned when n_cluster is 0.
    """

    def setUp(self):
        """
        Setup the temp folder for the target feature and cluster checkpoints.
        """
        self.test_dir = os.path.join(os.path.dirname(__file__),"temp_target_feats")
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """
        Remove the temp feature folder.
        """
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_cluster_creation(self):
        """
        Test the cluster calculation and saving behavior.

        This test ensures:
        - The cluster has the correct number of cluster centers.
        - The cluster file is saved to disk.
        - Original features are correctly loaded tf.target_feats.
        """
        dummy_feats = torch.rand(3000,1024)
        temp_feats_path = os.path.join(self.test_dir, "features.pt")
        torch.save(dummy_feats, temp_feats_path)

        for n_cluster in [32,128,256,2048]:
            tf = TargetFeats(target_features_path=temp_feats_path,n_cluster=n_cluster)
            cluster_feats = tf.get_cluster(save=True)

            self.assertTrue(torch.allclose(dummy_feats, tf.target_feats))

            self.assertEqual(cluster_feats.shape, (n_cluster, 1024))

            self.assertTrue(os.path.exists(tf.cluster_file_path))

    def test_cluster_reload(self):
        """
        Test loading of an already computed cluster.
        - This ensures that calling get_cluster a second time returns the same cluster without recomputing it.
        """
        dummy_feats = torch.rand(2000,1024)
        temp_feats_path = os.path.join(self.test_dir, "features.pt")
        torch.save(dummy_feats, temp_feats_path)

        for n_cluster in [16,64,256,1024]:
            tf = TargetFeats(target_features_path=temp_feats_path,n_cluster=n_cluster)
            
            clustered1 = tf.get_cluster(save=True)
            clustered2 = tf.get_cluster(save=False)
            self.assertTrue(torch.allclose(clustered1, clustered2)) 

            clustered3 = tf.get_cluster(save=True)
            self.assertTrue(torch.allclose(clustered1, clustered3)) 


    def test_no_clustering_returns_raw(self):
        """
        Test whether it returns the original features when n_cluster=0.
        - No cluster file is created.
        - Original features are returned.
        """
        dummy_feats = torch.rand(13490,1024)
        temp_feats_path = os.path.join(self.test_dir, "features.pt")
        torch.save(dummy_feats, temp_feats_path)

        tf = TargetFeats(target_features_path=temp_feats_path,n_cluster=0)
        cluster_feats = tf.get_cluster(save=False)

        self.assertFalse(os.path.exists(tf.cluster_file_path))
        self.assertTrue(torch.allclose(dummy_feats, cluster_feats))



if __name__ == "__main__":
    unittest.main()


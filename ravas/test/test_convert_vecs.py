import unittest
import torch

from stream_processing.models.knnvc.knnvc import convert_vecs, cosine_similarity


class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors(self):
        """
        Test case where both vectors are identical, the cosine similarity must be exactly 1.
        """
        a = torch.tensor([[1.0, 2.0, 3.0]])
        b = torch.tensor([[1.0, 2.0, 3.0]])
        sim = cosine_similarity(a, b)
        self.assertTrue(torch.allclose(sim, torch.tensor([[1.0]]), atol=1e-6))

    def test_orthogonal_vectors(self):
        """
        If two vectors are orthogonal (at 90 degrees),
        their cosine similarity must be 0..
        """
        source = torch.tensor([[1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0]])
        sim = cosine_similarity(source, target)
        self.assertTrue(torch.allclose(sim, torch.tensor([[0.0]]), atol=1e-6))

    def test_multiple_vectors_shape(self):
        """
        Cosine similarity is computed pairwise between all source and target vectors.

        If source has shape (N, D) and target has shape (M, D),
        the result must have shape (N, M).
        """
        source = torch.randn(4, 8) 
        target = torch.randn(6, 8)  
        sim = cosine_similarity(source, target)
        self.assertEqual(sim.shape, (4, 6))


class TestConvertVecs(unittest.TestCase):

    def test_single_neighbor(self):
        """
        With n_neighbors = 1, each source vector should be replaced
        by the single most similar target vector.

        Here the first target vector is an exact match,
        so the converted output must equal that target vector.
        """
        source = torch.tensor([[1.0, 0.0]])
        targets = torch.tensor([
            [1.0, 0.0],  
            [0.0, 1.0],  
        ])

        out = convert_vecs(source, targets, n_neighbors=1)
        expected = torch.tensor([[1.0, 0.0]])

        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_two_neighbors_average(self):
        """
        With n_neighbors = 2, the two most similar target vectors
        should be selected and averaged.

        The first two targets are closest to the source,
        so the output must equal their mean.
        """
        source = torch.tensor([[1.0, 0.0]])
        targets = torch.tensor([
            [1.0, 0.0],  # cosine similarity = 1.0
            [0.8, 0.2],  # cosine similarity ≈ 0.97
            [0.0, 1.0],  # cosine similarity = 0.0
        ])

        out = convert_vecs(source, targets, n_neighbors=2)

        # Expected average of the two closest target vectors
        expected = torch.tensor([[0.9, 0.1]])

        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_output_shape(self):
        """
        The converted vectors must preserve the temporal structure
        of the source embeddings.

        Therefore, output shape must match:
        (number of source vectors, feature dimension).
        """
        source = torch.randn(5, 16)   
        targets = torch.randn(20, 16) 

        out = convert_vecs(source, targets, n_neighbors=3)

        self.assertEqual(out.shape, source.shape)

    
    def test_deterministic_output(self):
        """
        Calling convert_vecs twice with the same inputs must
        produce exactly the same output.

        This ensures the k-NN conversion itself is deterministic.
        """
        torch.manual_seed(0)

        source = torch.randn(10, 32)
        targets = torch.randn(50, 32)
        n_neighbors = 5

        out1 = convert_vecs(source, targets, n_neighbors)
        out2 = convert_vecs(source, targets, n_neighbors)

        # Bit-exact equality is expected here
        self.assertTrue(
            torch.equal(out1, out2),
            "convert_vecs is not deterministic for identical inputs"
        )


if __name__ == "__main__":
    unittest.main()

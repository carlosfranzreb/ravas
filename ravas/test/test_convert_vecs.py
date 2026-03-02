import torch

from stream_processing.models.knnvc.knnvc import convert_vecs, cosine_similarity

"""
Test the `cosine_similarity` and `convert_vecs` functions.

This test suite verifies that:
- `cosine_similarity` computes correct similarity scores for known vector pairs.
- `cosine_similarity` returns tensors with the expected pairwise shape.
- `convert_vecs` correctly selects the nearest neighbors and averages them.
- `convert_vecs` preserves the shape of the source embeddings.
- `convert_vecs` produces deterministic outputs for identical inputs.
"""


def test_identical_vectors():
    """
    Test case where both vectors are identical,
    the cosine similarity must be exactly 1.
    """
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[1.0, 2.0, 3.0]])

    sim = cosine_similarity(a, b)

    assert torch.allclose(sim, torch.tensor([[1.0]]), atol=1e-6)


def test_orthogonal_vectors():
    """
    If two vectors are orthogonal (90 degrees),
    cosine similarity must be 0.
    """
    source = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0]])

    sim = cosine_similarity(source, target)

    assert torch.allclose(sim, torch.tensor([[0.0]]), atol=1e-6)


def test_multiple_vectors_shape():
    """
    If source has shape (N, D) and target has shape (M, D),
    the result must have shape (N, M).
    """
    source = torch.randn(4, 8)
    target = torch.randn(6, 8)

    sim = cosine_similarity(source, target)

    assert sim.shape == (4, 6)


# -------------------------
# convert_vecs Tests
# -------------------------


def test_single_neighbor():
    """
    With n_neighbors = 1, each source vector should be replaced
    by the single most similar target vector.
    """
    source = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    out = convert_vecs(source, targets, n_neighbors=1)
    expected = torch.tensor([[1.0, 0.0]])

    assert torch.allclose(out, expected, atol=1e-6)


def test_two_neighbors_average():
    """
    With n_neighbors = 2, the two most similar targets
    should be selected and averaged.
    """
    source = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor(
        [
            [1.0, 0.0],  # similarity = 1.0
            [0.8, 0.2],  # similarity ≈ 0.97
            [0.0, 1.0],  # similarity = 0.0
        ]
    )

    out = convert_vecs(source, targets, n_neighbors=2)

    expected = torch.tensor([[0.9, 0.1]])

    assert torch.allclose(out, expected, atol=1e-6)


def test_output_shape():
    """
    Output must preserve the shape of the source embeddings.
    """
    source = torch.randn(5, 16)
    targets = torch.randn(20, 16)

    out = convert_vecs(source, targets, n_neighbors=3)

    assert out.shape == source.shape


def test_deterministic_output():
    """
    Calling convert_vecs twice with the same inputs
    must produce exactly the same output.
    """
    torch.manual_seed(0)

    source = torch.randn(10, 32)
    targets = torch.randn(50, 32)
    n_neighbors = 5

    out1 = convert_vecs(source, targets, n_neighbors)
    out2 = convert_vecs(source, targets, n_neighbors)

    # Bit-exact equality expected
    assert torch.equal(out1, out2), (
        "convert_vecs is not deterministic for identical inputs"
    )

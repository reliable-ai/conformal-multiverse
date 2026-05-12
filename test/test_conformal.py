import numpy as np
from fairness_multiverse.conformal import compute_nc_scores


class TestComputeNcScores:
    """Tests for the compute_nc_scores function."""

    def test_binary_classification(self):
        """Test with binary classification (2 classes)."""
        probs = np.array([
            [0.7, 0.3],  # sample 0: high confidence in class 0
            [0.4, 0.6],  # sample 1: moderate confidence in class 1
            [0.9, 0.1],  # sample 2: very high confidence in class 0
        ])
        labels = np.array([0, 1, 0])

        expected = np.array([
            1.0 - 0.7,  # 0.3
            1.0 - 0.6,  # 0.4
            1.0 - 0.9,  # 0.1
        ])

        result = compute_nc_scores(probs, labels)
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiclass_classification(self):
        """Test with multi-class classification (more than 2 classes)."""
        probs = np.array([
            [0.6, 0.3, 0.1],  # sample 0: true label is class 0
            [0.2, 0.7, 0.1],  # sample 1: true label is class 1
            [0.1, 0.2, 0.7],  # sample 2: true label is class 2
            [0.5, 0.3, 0.2],  # sample 3: true label is class 0
        ])
        labels = np.array([0, 1, 2, 0])

        expected = np.array([
            1.0 - 0.6,  # 0.4
            1.0 - 0.7,  # 0.3
            1.0 - 0.7,  # 0.3
            1.0 - 0.5,  # 0.5
        ])

        result = compute_nc_scores(probs, labels)
        np.testing.assert_array_almost_equal(result, expected)

    def test_perfect_predictions(self):
        """Test when model assigns probability 1.0 to true class."""
        probs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        labels = np.array([0, 1])

        expected = np.array([0.0, 0.0])

        result = compute_nc_scores(probs, labels)
        np.testing.assert_array_almost_equal(result, expected)

    def test_worst_predictions(self):
        """Test when model assigns probability 0.0 to true class."""
        probs = np.array([
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        labels = np.array([0, 1])

        expected = np.array([1.0, 1.0])

        result = compute_nc_scores(probs, labels)
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_shape(self):
        """Test that output shape matches number of samples."""
        n_samples = 50
        n_classes = 4
        probs = np.random.rand(n_samples, n_classes)
        # Normalize to make valid probabilities
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, n_classes, n_samples)

        result = compute_nc_scores(probs, labels)

        assert result.shape == (n_samples,)
        assert result.dtype == np.float64

    def test_output_range(self):
        """Test that nonconformity scores are in [0, 1]."""
        probs = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.5, 0.5],
        ])
        labels = np.array([0, 1, 0])

        result = compute_nc_scores(probs, labels)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_single_sample(self):
        """Test with a single sample."""
        probs = np.array([[0.6, 0.4]])
        labels = np.array([1])

        expected = np.array([0.6])

        result = compute_nc_scores(probs, labels)
        np.testing.assert_array_almost_equal(result, expected)

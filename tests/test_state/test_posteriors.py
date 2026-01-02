"""Test the state posteriors module."""

import numpy as np

from latents.state import LatentsPosteriorStatic


# --------------------------
# Test LatentsPosteriorStatic class
# --------------------------
def test_latentsposteriorstatic_init_correct() -> None:
    """Test the correct instantiation of a LatentsPosteriorStatic class object."""
    # Create a LatentsPosteriorStatic object
    mean = np.ones((10, 20))
    cov = np.ones((10, 10))
    moment = np.ones((10, 10))
    X = LatentsPosteriorStatic(mean=mean, cov=cov, moment=moment)

    # Test attributes
    assert np.array_equal(X.mean, mean)
    assert np.array_equal(X.cov, cov)
    assert np.array_equal(X.moment, moment)

    # Test derived properties
    assert X.x_dim == 10
    assert X.n_samples == 20

    # Test repr
    assert (
        repr(X) == "LatentsPosteriorStatic(mean.shape=(10, 20), "
        "cov.shape=(10, 10), moment.shape=(10, 10))"
    )


def test_latentsposteriorstatic_copy() -> None:
    """Test the copy method of the LatentsPosteriorStatic class."""
    # Create a LatentsPosteriorStatic object
    mean = np.ones((10, 20))
    cov = np.ones((10, 10))
    moment = np.ones((10, 10))
    X = LatentsPosteriorStatic(mean=mean, cov=cov, moment=moment)

    # Test copy
    X_copy = X.copy()
    # Check that the attributes of the copy are the same as the original
    assert np.array_equal(X_copy.mean, X.mean)
    assert np.array_equal(X_copy.cov, X.cov)
    assert np.array_equal(X_copy.moment, X.moment)
    # Check that each attribute of the copy is a new array
    assert X_copy.mean.flags.owndata
    assert X_copy.cov.flags.owndata
    assert X_copy.moment.flags.owndata


def test_latentsposteriorstatic_clear() -> None:
    """Test the clear method of the LatentsPosteriorStatic class."""
    # Create a LatentsPosteriorStatic object
    mean = np.ones((10, 20))
    cov = np.ones((10, 10))
    moment = np.ones((10, 10))
    X = LatentsPosteriorStatic(mean=mean, cov=cov, moment=moment)

    # Test clear
    X.clear()
    for attr in vars(X):
        assert getattr(X, attr) is None

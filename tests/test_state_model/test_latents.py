"""Test the latents submodule."""

import numpy as np

from latents.state_model.latents import PosteriorLatentStatic


# --------------------------
# Test PosteriorLatentStatic class
# --------------------------
def test_posteriorlatentstatic_init_correct() -> None:
    """Test the correct instantation of a PosteriorLatentStatic class object."""
    # Create a PosteriorLatentStatic object
    mean = np.ones((10, 20))
    cov = np.ones((10, 10))
    moment = np.ones((10, 10))
    X = PosteriorLatentStatic(mean=mean, cov=cov, moment=moment)

    # Test attributes
    assert np.array_equal(X.mean, mean)
    assert np.array_equal(X.cov, cov)
    assert np.array_equal(X.moment, moment)

    # Test repr
    assert (
        repr(X) == "PosteriorLatentStatic(mean.shape=(10, 20), "
        "cov.shape=(10, 10), moment.shape=(10, 10))"
    )


def test_posteriorlatentstatic_copy() -> None:
    """Test the copy method of the PosteriorLatentStatic class."""
    # Create a PosteriorLatentStatic object
    mean = np.ones((10, 20))
    cov = np.ones((10, 10))
    moment = np.ones((10, 10))
    X = PosteriorLatentStatic(mean=mean, cov=cov, moment=moment)

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


def test_posteriorlatentstatic_clear() -> None:
    """Test the clear method of the PosteriorLatentStatic class."""
    # Create a PosteriorLatentStatic object
    mean = np.ones((10, 20))
    cov = np.ones((10, 10))
    moment = np.ones((10, 10))
    X = PosteriorLatentStatic(mean=mean, cov=cov, moment=moment)

    # Test clear
    X.clear()
    for attr in vars(X):
        assert getattr(X, attr) is None

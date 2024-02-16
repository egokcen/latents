"""Test the data_types module."""

import numpy as np
import pytest

from latents.gfa.data_types import (
    ObsData,
    PosteriorLatent,
)


# ------------------
# Test ObsData class
# ------------------
def test_obsdata_init_correct() -> None:
    """Test the correct instantation of an ObsData class object."""
    # Create an ObsData object
    data = np.ones((10, 20))
    dims = np.array([5, 5])
    Y = ObsData(data=data, dims=dims)

    # Test attributes
    assert np.array_equal(Y.data, data)
    assert np.array_equal(Y.dims, dims)

    # Test repr
    assert repr(Y) == "ObsData(data.shape=(10, 20), dims=[5 5])"


def test_obsdata_init_unspecified_attributes() -> None:
    """Test for correct exceptions when ObsData attributes are unspecified."""
    data = np.ones((10, 20))
    dims = np.array([5, 5])

    # Test unspecified data or dims
    with pytest.raises(TypeError, match="data must be a numpy.ndarray."):
        ObsData(data=None, dims=None)
    with pytest.raises(TypeError, match="data must be a numpy.ndarray."):
        ObsData(data=None, dims=dims)
    with pytest.raises(TypeError, match="dims must be a numpy.ndarray."):
        ObsData(data=data, dims=None)


def test_obsdata_init_mismatched_dims() -> None:
    """Test for correct exceptions when ObsData data and dims are mismatched."""
    data = np.ones((10, 20))
    dims = np.array([5, 6])
    with pytest.raises(
        ValueError, match="The sum of dims must equal the number of rows in data."
    ):
        ObsData(data=data, dims=dims)


def test_obsdata_get_groups() -> None:
    """Test the get_groups method of the ObsData class."""
    # Create an ObsData object
    Y = ObsData(
        data=np.arange(90).reshape(9, 10).copy(),
        dims=np.array([2, 3, 4]),
    )

    # Test get_groups
    groups = Y.get_groups()
    # Check for the correct number of groups
    assert len(groups) == Y.dims.shape[0]
    # Check that the elements of each group are correct
    assert np.array_equal(groups[0], Y.data[:2, :])
    assert np.array_equal(groups[1], Y.data[2:5, :])
    assert np.array_equal(groups[2], Y.data[5:, :])
    # Check that each group is merely a view of the original data
    for group_idx in range(len(groups)):
        assert not groups[group_idx].flags.owndata  # Data is not owned by the group
        assert groups[group_idx].base is Y.data  # Data is a view of the original data


# --------------------------
# Test PosteriorLatent class
# --------------------------
def test_posteriorlatent_init_correct() -> None:
    """Test the correct instantation of a PosteriorLatent class object."""
    # Create a PosteriorLatent object
    mean = np.ones((10, 20))
    cov = np.ones((10, 10))
    moment = np.ones((10, 10))
    X = PosteriorLatent(mean=mean, cov=cov, moment=moment)

    # Test attributes
    assert np.array_equal(X.mean, mean)
    assert np.array_equal(X.cov, cov)
    assert np.array_equal(X.moment, moment)

    # Test repr
    assert (
        repr(X) == "PosteriorLatent(mean.shape=(10, 20), "
        "cov.shape=(10, 10), moment.shape=(10, 10))"
    )


def test_posteriorlatent_copy() -> None:
    """Test the copy method of the PosteriorLatent class."""
    # Create a PosteriorLatent object
    mean = np.ones((10, 20))
    cov = np.ones((10, 10))
    moment = np.ones((10, 10))
    X = PosteriorLatent(mean=mean, cov=cov, moment=moment)

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


def test_posteriorlatent_clear() -> None:
    """Test the clear method of the PosteriorLatent class."""
    # Create a PosteriorLatent object
    mean = np.ones((10, 20))
    cov = np.ones((10, 10))
    moment = np.ones((10, 10))
    X = PosteriorLatent(mean=mean, cov=cov, moment=moment)

    # Test clear
    X.clear()
    for attr in vars(X):
        assert getattr(X, attr) is None

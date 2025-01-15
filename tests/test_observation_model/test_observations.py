"""Test the observations submodule."""

import numpy as np
import pytest

from latents.observation_model.observations import ObsStatic


# ------------------
# Test ObsStatic class
# ------------------
def test_obsstaticinit_correct() -> None:
    """Test the correct instantation of an ObsStatic class object."""
    # Create an ObsStatic object
    data = np.ones((10, 20))
    dims = np.array([5, 5])
    Y = ObsStatic(data=data, dims=dims)

    # Test attributes
    assert np.array_equal(Y.data, data)
    assert np.array_equal(Y.dims, dims)

    # Test repr
    assert repr(Y) == "ObsStatic(data.shape=(10, 20), dims=[5 5])"


def test_obsstatic_init_unspecified_attributes() -> None:
    """Test for correct exceptions when ObsStatic attributes are unspecified."""
    data = np.ones((10, 20))
    dims = np.array([5, 5])

    # Test unspecified data or dims
    with pytest.raises(TypeError, match="data must be a numpy.ndarray."):
        ObsStatic(data=None, dims=None)
    with pytest.raises(TypeError, match="data must be a numpy.ndarray."):
        ObsStatic(data=None, dims=dims)
    with pytest.raises(TypeError, match="dims must be a numpy.ndarray."):
        ObsStatic(data=data, dims=None)


def test_obsstatic_init_mismatched_dims() -> None:
    """Test for correct exceptions when ObsStatic data and dims are mismatched."""
    data = np.ones((10, 20))
    dims = np.array([5, 6])
    with pytest.raises(
        ValueError, match="The sum of dims must equal the number of rows in data."
    ):
        ObsStatic(data=data, dims=dims)


def test_obsstatic_get_groups() -> None:
    """Test the get_groups method of the ObsStatic class."""
    # Create an ObsStatic object
    Y = ObsStatic(
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

"""Test the data module."""

import numpy as np
import pytest

from latents.data import ObsStatic


class TestObsStatic:
    """Tests for ObsStatic data container."""

    def test_init_correct(self) -> None:
        """Correct instantiation sets attributes and repr."""
        data = np.ones((10, 20))
        dims = np.array([5, 5])
        Y = ObsStatic(data=data, dims=dims)

        assert np.array_equal(Y.data, data)
        assert np.array_equal(Y.dims, dims)
        assert repr(Y) == "ObsStatic(data.shape=(10, 20), dims=[5 5])"

    def test_init_unspecified_attributes(self) -> None:
        """Raises TypeError when data or dims are None."""
        data = np.ones((10, 20))
        dims = np.array([5, 5])

        with pytest.raises(TypeError, match=r"data must be a numpy\.ndarray\."):
            ObsStatic(data=None, dims=None)
        with pytest.raises(TypeError, match=r"data must be a numpy\.ndarray\."):
            ObsStatic(data=None, dims=dims)
        with pytest.raises(TypeError, match=r"dims must be a numpy\.ndarray\."):
            ObsStatic(data=data, dims=None)

    def test_init_mismatched_dims(self) -> None:
        """Raises ValueError when dims don't sum to data rows."""
        data = np.ones((10, 20))
        dims = np.array([5, 6])
        with pytest.raises(
            ValueError, match=r"The sum of dims must equal the number of rows in data\."
        ):
            ObsStatic(data=data, dims=dims)

    def test_get_groups(self) -> None:
        """get_groups returns correct views into the data array."""
        Y = ObsStatic(
            data=np.arange(90).reshape(9, 10).copy(),
            dims=np.array([2, 3, 4]),
        )

        groups = Y.get_groups()
        assert len(groups) == Y.dims.shape[0]
        assert np.array_equal(groups[0], Y.data[:2, :])
        assert np.array_equal(groups[1], Y.data[2:5, :])
        assert np.array_equal(groups[2], Y.data[5:, :])
        # Each group is a view, not a copy
        for group_idx in range(len(groups)):
            assert not groups[group_idx].flags.owndata
            assert groups[group_idx].base is Y.data

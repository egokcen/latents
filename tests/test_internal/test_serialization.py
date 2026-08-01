"""Test safetensors serialization helpers."""

from __future__ import annotations

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file

from latents._internal.serialization import as_contiguous, save_tensors


class TestAsContiguous:
    """Tests for as_contiguous()."""

    def test_passes_through_contiguous(self):
        """C-contiguous arrays are returned unchanged, without a copy."""
        a = np.arange(6.0).reshape(2, 3)
        assert as_contiguous(a) is a

    def test_copies_fortran_order(self):
        """F-contiguous arrays are copied to C order, preserving values."""
        a = np.arange(6.0).reshape(2, 3).T  # (3, 2), F-contiguous view
        out = as_contiguous(a)
        assert out.flags.c_contiguous
        np.testing.assert_array_equal(out, a)

    def test_copies_strided_view(self):
        """Strided (non-dense) views are materialized, preserving values."""
        a = np.arange(10.0)[::2]
        out = as_contiguous(a)
        assert out.flags.c_contiguous
        np.testing.assert_array_equal(out, a)

    def test_preserves_zero_dim_shape(self):
        """0-d arrays keep shape (), which np.ascontiguousarray would promote."""
        a = np.array(5.0)
        out = as_contiguous(a)
        assert out.shape == ()
        # Guard against a naive unconditional np.ascontiguousarray() call
        assert np.ascontiguousarray(a).shape == (1,)

    def test_preserves_dtype(self):
        """Normalization does not change dtype."""
        a = np.arange(6, dtype=np.int32).reshape(2, 3).T
        assert as_contiguous(a).dtype == np.int32


class TestSaveTensors:
    """Tests for save_tensors()."""

    @pytest.mark.parametrize(
        ("name", "array"),
        [
            ("c_order", np.arange(12.0).reshape(3, 4)),
            ("f_order", np.arange(12.0).reshape(3, 4).T),
            ("strided", np.arange(20.0)[::2]),
            ("zero_dim", np.array(7.0)),
            ("int_f_order", np.arange(12, dtype=np.int64).reshape(3, 4).T),
        ],
    )
    def test_round_trip(self, name, array, tmp_path):
        """Arrays of any memory layout round-trip exactly.

        Regression test: safetensors >=0.8.0 serializes from the raw buffer
        pointer, so a non-contiguous array is silently written in the wrong
        element order unless normalized first.
        """
        path = tmp_path / f"{name}.safetensors"
        save_tensors({name: array}, path, {})

        loaded = load_file(path)[name]
        assert loaded.shape == array.shape
        assert loaded.dtype == array.dtype
        np.testing.assert_array_equal(loaded, array)

    def test_does_not_mutate_input(self, tmp_path):
        """Normalization leaves the caller's arrays untouched."""
        a = np.arange(12.0).reshape(3, 4).T
        original = a.copy()
        save_tensors({"a": a}, tmp_path / "x.safetensors", {})
        np.testing.assert_array_equal(a, original)
        assert a.flags.f_contiguous  # layout unchanged

    def test_metadata_preserved(self, tmp_path):
        """String metadata survives the wrapper."""
        path = tmp_path / "meta.safetensors"
        save_tensors({"a": np.arange(4.0)}, path, {"key": "value"})

        with safe_open(path, framework="numpy") as f:
            assert f.metadata() == {"key": "value"}

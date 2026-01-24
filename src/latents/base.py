"""General-purpose base classes."""

from __future__ import annotations

import numpy as np


class ArrayContainer:
    """Base class for containers holding `numpy.ndarray` attributes.

    Provides common functionality for deep copying and clearing array attributes,
    plus a custom ``__repr__`` that shows shapes instead of full values.

    Notes
    -----
    This is an abstract base class with no attributes of its own. Subclasses
    should define their own ``__init__`` that stores arrays as instance attributes.
    The ``copy()`` and ``clear()`` methods operate on all instance attributes.
    """

    def __repr__(self) -> str:
        # If array attributes are specified, then display their shapes, rather
        # than their full values
        attr_reprs = []
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, np.ndarray):
                attr_reprs.append(f"{attr_name}.shape={attr_value.shape}")
            else:
                attr_reprs.append(f"{attr_name}={attr_value}")

        return type(self).__name__ + "(" + ", ".join(attr_reprs) + ")"

    def copy(self) -> ArrayContainer:
        """Return a deep copy of self.

        Returns
        -------
        ArrayContainer
            A deep copy of self.
        """
        # If array attributes are specified, then create copies of them, and
        # create a new class instance with those copies
        new_attrs = {}
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, np.ndarray):
                new_attrs[attr_name] = attr_value.copy()
            else:
                new_attrs[attr_name] = attr_value
        return self.__class__(**new_attrs)

    def clear(self) -> None:
        """Set all attributes to None."""
        for attr_name in vars(self):
            setattr(self, attr_name, None)

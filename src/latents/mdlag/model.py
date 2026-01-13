"""mDLAGModel class for delayed latents across multiple groups."""

from __future__ import annotations

_NOT_IMPLEMENTED_MSG = "mDLAG not yet implemented"


class mDLAGModel:
    """High-level interface for mDLAG."""

    def __init__(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def fit(self):
        """Fit model to data via variational inference."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def infer_latents(self):
        """Infer latent posterior for new data given fitted parameters."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def save(self, filename: str) -> None:
        """Save model to a JSON file."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    @staticmethod
    def load(filename: str):
        """Load model from a JSON file."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

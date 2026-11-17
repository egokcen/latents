"""pPCAModel class for probabilistic Principal Component Analysis."""

from __future__ import annotations

import os

_NOT_IMPLEMENTED_MSG = "pPCA not yet implemented"


class pPCAModel:
    """High-level interface for pPCA.

    Raises
    ------
    NotImplementedError
        Always raised; this class is a placeholder for future implementation.
    """

    def __init__(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def fit(self):
        """Fit model to data via Expectation-Maximization.

        Raises
        ------
        NotImplementedError
            pPCA is not yet implemented.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def infer_latents(self):
        """Infer latent posterior for new data given fitted parameters.

        Raises
        ------
        NotImplementedError
            pPCA is not yet implemented.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save model to a safetensors file.

        Parameters
        ----------
        path : str or PathLike
            Output file path (conventionally ends in .safetensors).

        Raises
        ------
        NotImplementedError
            pPCA is not yet implemented.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> pPCAModel:
        """Load model from a safetensors file.

        Parameters
        ----------
        path : str or PathLike
            Path to .safetensors file.

        Returns
        -------
        pPCAModel
            Loaded model, ready for inference or continued fitting.

        Raises
        ------
        NotImplementedError
            pPCA is not yet implemented.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

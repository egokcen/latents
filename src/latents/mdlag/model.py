"""mDLAGModel class for delayed latents across multiple groups."""

from __future__ import annotations

_NOT_IMPLEMENTED_MSG = "mDLAG not yet implemented"


class mDLAGModel:
    """High-level interface for mDLAG.

    Stub for future implementation. Will provide a similar interface to
    :class:`GFAModel`.

    Raises
    ------
    NotImplementedError
        Always raised; this class is a placeholder for future implementation.
    """

    def __init__(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def fit(self):
        """Fit model to data via variational inference.

        Raises
        ------
        NotImplementedError
            mDLAG is not yet implemented.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def infer_latents(self):
        """Infer latent posterior for new data given fitted parameters.

        Raises
        ------
        NotImplementedError
            mDLAG is not yet implemented.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def save(self, filename: str) -> None:
        """Save model to a JSON file.

        Parameters
        ----------
        filename : str
            Path to output file.

        Raises
        ------
        NotImplementedError
            mDLAG is not yet implemented.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    @staticmethod
    def load(filename: str):
        """Load model from a JSON file.

        Parameters
        ----------
        filename : str
            Path to input file.

        Returns
        -------
        mDLAGModel
            Loaded model instance.

        Raises
        ------
        NotImplementedError
            mDLAG is not yet implemented.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

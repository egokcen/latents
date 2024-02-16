"""Test imports and the overall structure of the latents package."""

from importlib.metadata import version as get_version


def test_latents_import():
    """Test that the latents package can be imported."""
    import latents

    # Check that the package is in the current namespace
    assert latents.__name__ in dir()


def test_latents_version():
    """Test that the version of the latents package can be retrieved."""
    import latents

    assert latents.__version__ == get_version("latents")


def test_latents_namespace():
    """Test that the latents package has the expected namespace."""
    import latents

    # Check that subpackages are in the namespace of latents
    assert "gfa" in dir(latents)
    assert "mdlag" in dir(latents)


def test_latents_import_subpackages():
    """Test the 'from latents import' mechanism."""
    from latents import (
        gfa,  # noqa: F401
        mdlag,  # noqa: F401
    )

    # Check that subpackages are in the current namespace
    assert "gfa" in dir()
    assert "mdlag" in dir()


def test_gfa_namespace():
    """Test that the gfa subpackage has the expected namespace."""
    from latents import gfa

    # Check that submodules are in the namespace of gfa
    assert "core" in dir(gfa)
    assert "data_types" in dir(gfa)
    assert "descriptive_stats" in dir(gfa)
    assert "plotting" in dir(gfa)
    assert "simulation" in dir(gfa)

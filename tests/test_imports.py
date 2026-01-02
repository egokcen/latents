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
    assert "observation_model" in dir(latents)
    assert "state_model" in dir(latents)


def test_latents_import_subpackages():
    """Test the 'from latents import' mechanism."""
    from latents import (
        gfa,  # noqa: F401
        mdlag,  # noqa: F401
        observation_model,  # noqa: F401
        state_model,  # noqa: F401
    )

    # Check that subpackages are in the current namespace
    assert "gfa" in dir()
    assert "mdlag" in dir()
    assert "observation_model" in dir()
    assert "state_model" in dir()


def test_gfa_namespace():
    """Test that the gfa subpackage has the expected namespace."""
    from latents import gfa

    # Check that submodules are in the namespace of gfa
    assert "core" in dir(gfa)
    assert "data_types" in dir(gfa)
    assert "descriptive_stats" in dir(gfa)
    assert "simulation" in dir(gfa)

    # Check other assets exposed to the user
    assert "GFAModel" in dir(gfa)


def test_mdlag_namespace():
    """Test that the mdlag subpackage has the expected namespace."""
    from latents import mdlag

    # Check that submodules are in the namespace of mdlag
    assert "core" in dir(mdlag)
    assert "data_types" in dir(mdlag)
    assert "descriptive_stats" in dir(mdlag)
    assert "simulation" in dir(mdlag)

    # Check other assets exposed to the user
    assert "mDLAGModel" in dir(mdlag)


def test_observation_model_namespace():
    """Test that the observation_model subpackage has the expected namespace."""
    from latents import observation_model

    # Check that submodules are in the namespace of observation_model
    assert "observations" in dir(observation_model)
    assert "probabilistic" in dir(observation_model)

    # Check other assets exposed to the user
    assert "ObsStatic" in dir(observation_model.observations)
    assert "ObsTimeSeries" in dir(observation_model.observations)


def test_state_model_namespace():
    """Test that the state_model subpackage has the expected namespace."""
    from latents import state_model

    # Check that submodules are in the namespace of state_model
    assert "latents" in dir(state_model)

    # Check other assets exposed to the user
    assert "PosteriorLatentDelayed" in dir(state_model.latents)
    assert "PosteriorLatentStatic" in dir(state_model.latents)

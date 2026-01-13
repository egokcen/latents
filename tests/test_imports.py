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

    # Check that subpackages/modules are in the namespace of latents
    assert "data" in dir(latents)
    assert "gfa" in dir(latents)
    assert "mdlag" in dir(latents)
    assert "observation" in dir(latents)
    assert "state" in dir(latents)


def test_latents_import_subpackages():
    """Test the 'from latents import' mechanism."""
    from latents import (
        data,  # noqa: F401
        gfa,  # noqa: F401
        mdlag,  # noqa: F401
        observation,  # noqa: F401
        state,  # noqa: F401
    )

    # Check that subpackages/modules are in the current namespace
    assert "data" in dir()
    assert "gfa" in dir()
    assert "mdlag" in dir()
    assert "observation" in dir()
    assert "state" in dir()


def test_gfa_namespace():
    """Test that the gfa subpackage has the expected namespace."""
    from latents import gfa

    # Check primary exports
    assert "GFAModel" in dir(gfa)
    assert "GFAFitConfig" in dir(gfa)


def test_mdlag_namespace():
    """Test that the mdlag subpackage has the expected namespace."""
    from latents import mdlag

    # Check primary exports
    assert "mDLAGModel" in dir(mdlag)
    assert "mDLAGFitConfig" in dir(mdlag)


def test_data_namespace():
    """Test that the data module has the expected namespace."""
    from latents import data

    # Check that data classes are in the namespace of data
    assert "ObsStatic" in dir(data)
    assert "ObsTimeSeries" in dir(data)


def test_observation_namespace():
    """Test that the observation subpackage has the expected namespace."""
    from latents import observation

    # Check posterior classes
    assert "LoadingPosterior" in dir(observation)
    assert "ARDPosterior" in dir(observation)
    assert "ObsMeanPosterior" in dir(observation)
    assert "ObsPrecPosterior" in dir(observation)
    assert "ObsParamsPosterior" in dir(observation)

    # Check hyperprior classes
    assert "ObsParamsHyperPrior" in dir(observation)
    assert "ObsParamsHyperPriorStructured" in dir(observation)
    assert "ObsParamsPrior" in dir(observation)

    # Check realization classes and utilities
    assert "ObsParamsRealization" in dir(observation)
    assert "ObsParamsPoint" in dir(observation)
    assert "adjust_snr" in dir(observation)


def test_state_namespace():
    """Test that the state subpackage has the expected namespace."""
    from latents import state

    # Check posterior classes
    assert "LatentsPosteriorStatic" in dir(state)
    assert "LatentsPosteriorTimeSeries" in dir(state)
    assert "LatentsPosteriorDelayed" in dir(state)

    # Check prior classes
    assert "LatentsPriorStatic" in dir(state)
    assert "LatentsPriorGP" in dir(state)
    assert "LatentsHyperPriorGP" in dir(state)

    # Check realization classes
    assert "LatentsRealization" in dir(state)


def test_plotting_namespace():
    """Test that the plotting subpackage has the expected namespace."""
    from latents import plotting

    # Check visualization functions
    assert "hinton_diagram" in dir(plotting)
    assert "plot_dimensionalities" in dir(plotting)
    assert "plot_var_exp" in dir(plotting)
    assert "plot_dims_pairs" in dir(plotting)
    assert "plot_var_exp_pairs" in dir(plotting)

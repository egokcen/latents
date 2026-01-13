# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Documentation hosted on [Read the Docs](https://latents.readthedocs.io)
- Interactive examples gallery using sphinx-gallery
- Python 3.14 support
- Optional `notebook` extra for Jupyter support (`pip install latents[notebook]`)
- `GFAFitConfig` frozen dataclass for immutable GFA fitting configuration
- New `observation/` subpackage with probabilistic hierarchy:
  - `ObsParamsHyperPrior` and `ObsParamsHyperPriorStructured` for hyperprior configuration
  - `ObsParamsPrior` with `sample()` method for forward sampling from priors
  - `ObsParamsRealization` and `ObsParamsPoint` for parameter value containers
  - `adjust_snr()` utility for signal-to-noise ratio adjustment
  - Posterior classes: `LoadingPosterior`, `ARDPosterior`, `ObsMeanPosterior`, `ObsPrecPosterior`, `ObsParamsPosterior`
- New `state/` subpackage with probabilistic hierarchy:
  - `LatentsPriorStatic` with `sample()` method for forward sampling
  - `LatentsPosteriorStatic` with `sample()` and `posterior_mean` for posterior inference
  - `LatentsRealization` for sampled latent values
  - GP prior/posterior stubs for future GPFA/mDLAG support
- New `plotting/` subpackage for visualization utilities:
  - `hinton_diagram()` for matrix visualization (renamed from `hinton()`)
  - `plot_dimensionalities()`, `plot_var_exp()`, `plot_dims_pairs()`, `plot_var_exp_pairs()` for observation model results
- `GFAModel.recompute_latents(Y)` to restore latents after loading a model saved with `save_x=False`
- `GFAModel.recompute_loadings(Y)` to restore loading covariances after loading a model saved with `save_c_cov=False`

### Changed

- GFA variational inference iteration order changed from d → X → C → α → φ to d → C → α → φ → X. This enables exact reconstruction of cleared latents from saved observation parameters, allowing `resume_fit()` and `compute_lower_bound()` to work with `save_x=False`.
- Internal utilities reorganized into `_core/` subpackage (base classes, fitting infrastructure, numerics)
- Observation model probabilistic components reorganized into `observation/` subpackage
- Observation data containers (`ObsStatic`, `ObsTimeSeries`) moved to `latents.data` module
- State model classes reorganized into `state/` subpackage
- `simulate()` now returns `LatentsRealization` instead of raw ndarray
- Minimum Python version raised from 3.9 to 3.10
- Jupyter dependencies (`jupyter`, `ipywidgets`) moved from core to optional extra
- `GFAModel` now accepts `config` parameter in constructor instead of using `fit_args.set_args()`
- Configuration uses snake_case naming (e.g., `prune_x` instead of `prune_X`)
- Methods with `in_place=True` now return `self` instead of `None`. Affected methods: `get_subset_dims`, `compute_moment`, `compute_mean`, and all `infer_*` functions

### Removed

- Python 3.9 support
- Unused `pandas` dependency
- `GFAFitArgs` class (replaced by `GFAFitConfig`)
- `HyperPriorParams` class (replaced by `ObsParamsHyperPrior` and `ObsParamsHyperPriorStructured`)
- `observation_model/` subpackage (probabilistic classes moved to `observation/`, data containers moved to `data`)
- `state_model/` subpackage (classes moved to `state/` subpackage)
- `StateParamsStatic` class (functionality absorbed into `LatentsPosteriorStatic`)
- `plotting.py` module (replaced by `plotting/` subpackage)
- `ObsParamsPosterior` plotting methods (moved to `plotting/` subpackage as standalone functions)

## [0.0.4] - 2024-10-31

### Added

- mDLAG (Delayed Latents Across Multiple Groups) module shell
- Contributing guide and community documentation
- Code of Conduct (Contributor Covenant 2.1)
- Issue templates and pull request template
- GitHub Actions workflows for testing and documentation

### Changed

- Refactored package structure for extensibility

## [0.0.3] - 2024-02-16

### Added

- Group Factor Analysis (GFA) implementation
- Initial documentation and Sphinx setup
- Unit tests for GFA module

## [0.0.2] - 2024-02-15

### Added

- Initial project structure
- Package scaffolding with setuptools

[Unreleased]: https://github.com/egokcen/latents/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/egokcen/latents/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/egokcen/latents/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/egokcen/latents/releases/tag/v0.0.2

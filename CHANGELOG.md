# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Documentation hosted on [Read the Docs](https://latents.readthedocs.io)
- Interactive examples gallery using sphinx-gallery
- GFA benchmark suite for runtime scaling, parameter recovery, and dimensionality recovery (`benchmarks/`)
- Benchmark developer guide and Sphinx-gallery integration for benchmark results
- Python 3.14 support
- `GFAFitConfig` frozen dataclass for immutable GFA fitting configuration
- `GFASimConfig` frozen dataclass for immutable GFA simulation configuration
- `GFASimulationResult` dataclass bundling simulation outputs (config, hyperprior, obs_params, latents, observations)
- Simulation persistence functions: `save_simulation()`, `load_simulation()` for full snapshots; `save_simulation_recipe()`, `load_simulation_recipe()` for reproducible recipes
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
- Callback system for model fitting (`latents.callbacks`):
  - `ProgressCallback` for tqdm progress bar during fitting
  - `LoggingCallback` for structured logging to the `latents` logger
  - `CheckpointCallback` for periodic model checkpointing with safetensors format
- `GFAFitContext` dataclass providing model state to callbacks
- New `latents.base` module with `ArrayContainer` base class for array-holding containers
- New `latents.tracking` module with base `FitTracker` and `FitFlags` classes
- New `latents.mdlag.tracking` module with `mDLAGFitTracker` and `mDLAGFitFlags` stubs
- Package root now exports `base`, `callbacks`, and `tracking` modules

### Changed

- GFA fitting loop optimized to reduce temporary memory allocations: use `np.einsum` to avoid broadcast temporaries, pre-compute `Y_sum` and `Y_centered` to eliminate redundant `(y_dim, n_samples)` arrays. 10–65% faster per-iteration runtime depending on problem dimensions.
- GFA variational inference iteration order changed from d → X → C → α → φ to d → C → α → φ → X. This enables exact reconstruction of cleared latents from saved observation parameters, allowing `resume_fit()` and `compute_lower_bound()` to work with `save_x=False`.
- Internal utilities reorganized into `_internal/` subpackage (numerics, logging)
- Observation model probabilistic components reorganized into `observation/` subpackage
- Observation data containers (`ObsStatic`, `ObsTimeSeries`) moved to `latents.data` module
- State model classes reorganized into `state/` subpackage
- `simulate()` signature changed to `simulate(config, hyperprior)` returning `GFASimulationResult` instead of tuple
- Minimum Python version raised from 3.9 to 3.10
- `GFAModel` now accepts `config` parameter in constructor instead of using `fit_args.set_args()`
- Configuration uses snake_case naming (e.g., `prune_x` instead of `prune_X`)
- Methods with `in_place=True` now return `self` instead of `None`. Affected methods: `get_subset_dims`, `compute_moment`, `compute_mean`, and all `infer_*` functions
- `LatentsRealization.X` renamed to `LatentsRealization.data` for API consistency with other realization classes
- `random_seed` parameter in `GFAFitConfig` and `GFASimConfig` now accepts `int | Sequence[int] | None` (previously `int | None`) for structured seeding in parallel experiments

### Removed

- Python 3.9 support
- Unused `pandas` dependency
- `GFAFitArgs` class (replaced by `GFAFitConfig`)
- `verbose` parameter from `GFAFitConfig` (use `ProgressCallback` instead)
- `HyperPriorParams` class (replaced by `ObsParamsHyperPrior` and `ObsParamsHyperPriorStructured`)
- `observation_model/` subpackage (probabilistic classes moved to `observation/`, data containers moved to `data`)
- `state_model/` subpackage (classes moved to `state/` subpackage)
- `StateParamsStatic` class (functionality absorbed into `LatentsPosteriorStatic`)
- `plotting.py` module (replaced by `plotting/` subpackage)
- `ObsParamsPosterior` plotting methods (moved to `plotting/` subpackage as standalone functions)
- `notebook` optional extra (`pip install latents[notebook]`); users should install Jupyter separately if needed

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

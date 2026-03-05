# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Documentation hosted on [Read the Docs](https://latents.readthedocs.io) with interactive examples gallery and benchmark results
- Python 3.14 support
- `observation/` subpackage with full probabilistic hierarchy: hyperpriors, priors with `sample()` methods, posterior classes, and realization containers
- `state/` subpackage with probabilistic hierarchy: `LatentsPriorStatic`, `LatentsPosteriorStatic` with `sample()` and `posterior_mean`, and `LatentsRealization`
- `plotting/` subpackage: `hinton_diagram()` for matrix visualization, `plot_dimensionalities()`, `plot_var_exp()`, and pair-plot variants for observation model results
- `GFAFitConfig` and `GFASimConfig` frozen dataclasses for immutable configuration
- `GFASimulationResult` dataclass bundling simulation outputs with persistence functions (`save_simulation()`, `load_simulation()`, `save_simulation_recipe()`, `load_simulation_recipe()`)
- Callback system for model fitting: `ProgressCallback` (tqdm), `LoggingCallback` (structured logging), `CheckpointCallback` (periodic safetensors checkpoints)
- `GFAModel.recompute_latents(Y)` and `GFAModel.recompute_loadings(Y)` to restore cleared parameters after loading saved models

### Changed

- Package reorganized: `observation_model/` → `observation/` and `latents.data`, `state_model/` → `state/`, `plotting.py` → `plotting/`, internal utilities → `_internal/`
- GFA fitting loop optimized to reduce temporary memory allocations via `np.einsum` and pre-computed intermediate arrays. 10–65% faster per-iteration runtime depending on problem dimensions.
- GFA inference iteration order changed from d → X → C → α → φ to d → C → α → φ → X, enabling exact latent reconstruction from saved observation parameters (`resume_fit()` and `compute_lower_bound()` now work with `save_x=False`)
- `simulate()` signature changed to `simulate(config, hyperprior)` returning `GFASimulationResult`
- Methods with `in_place=True` now return `self` instead of `None`
- `random_seed` now accepts `int | Sequence[int] | None` for structured seeding in parallel experiments
- Minimum Python version raised from 3.9 to 3.10

### Removed

- Python 3.9 support
- `pandas` dependency
- `GFAFitArgs` class (replaced by `GFAFitConfig`), `HyperPriorParams` (replaced by `ObsParamsHyperPrior`), `StateParamsStatic` (absorbed into `LatentsPosteriorStatic`)
- `verbose` parameter from fitting config (use `ProgressCallback` instead)
- `notebook` optional extra; install Jupyter separately if needed

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

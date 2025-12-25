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

### Changed

- Minimum Python version raised from 3.9 to 3.10
- Jupyter dependencies (`jupyter`, `ipywidgets`) moved from core to optional extra

### Removed

- Python 3.9 support
- Unused `pandas` dependency

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

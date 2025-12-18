# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Migrated build system from setuptools to Hatchling with hatch-vcs
- Migrated package manager from pip to uv
- Updated Python support from 3.9-3.13 to 3.10-3.14
- Moved `notebook` from core dependencies to optional extra (`pip install latents[notebook]`)
- Removed unused `pandas` dependency
- Added PEP 735 dependency groups for development dependencies

[Unreleased]: https://github.com/egokcen/latents/compare/v0.0.5...HEAD

# AGENTS.md

This file provides shared guidance to artificial intelligence (AI) coding agents working
in this repository.

## Project overview

Latents is a Python library for latent variable modeling and dimensionality reduction, emphasizing linear, probabilistic methods.

**Supported methods:**
- Group Factor Analysis (GFA) - `latents.gfa`
- Delayed Latents Across Multiple Groups (mDLAG) - `latents.mdlag` (under development)

## Build and development commands

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```sh
# Install all dependencies (creates .venv automatically)
uv sync --all-groups

# Run all tests with coverage
uv run pytest

# Run a single test file
uv run pytest tests/test_gfa/test_inference.py

# Run a specific test
uv run pytest tests/test_gfa/test_inference.py::test_fit

# Fast tests only (skip model fitting)
uv run pytest -m "not fit"

# Run linting (pre-commit hooks)
uv run ruff check  # Linting
uv run ruff format  # Formatting
uv run pre-commit run --all-files  # All pre-commit checks (includes ruff)

# Git commit (pre-commit hooks require uv run)
uv run git commit -m "your message"

# Build documentation (warnings are errors)
uv run sphinx-build -W -b html docs/source docs/_build/html

# Build package (sdist and wheel)
uv build
```

### Testing on multiple Python versions

```sh
# Switch to a different Python version (re-syncs dependencies)
uv sync --python 3.10 --all-groups
uv run pytest

uv sync --python 3.14 --all-groups
uv run pytest
```

## AI agent infrastructure

- `AGENTS.md` is the canonical shared instruction file. `CLAUDE.md` imports it.
- `.agents/skills/` is the canonical home for reusable project skills. Claude skill
  locations link to the corresponding canonical directories.
- `.claude/agents/` and `.codex/agents/` contain only thin, host-specific adapters.
  Keep reusable procedures and report formats in skills rather than copying them into
  agent definitions.
- Audit agents return findings to the requesting conversation. Persist an audit report
  only when the user explicitly requests a file.
- Claude audit adapters use `dontAsk` with only `Read`, `Grep`, and `Glob`. If useful
  validation requires shell access or would write artifacts, ask the orchestrator to
  run it.

## Code architecture

### Package structure

```text
src/latents/
├── observation/  # Observation-model components by probabilistic level
├── state/        # Latent-state components by probabilistic level
├── plotting/     # Visualization utilities
├── gfa/          # Group Factor Analysis
├── mdlag/        # Delayed Latents Across Multiple Groups
└── _internal/    # Private infrastructure and numerical utilities
```

The filesystem is the source of truth for the exact module inventory. Inspect
`src/latents/` rather than expanding this map with individual filenames.
Top-level modules provide shared data containers, callbacks, fit tracking, and base
classes.

### Core design patterns

1. **Model wrapper classes** (`GFAModel`, `mDLAGModel`): High-level interface storing posteriors, tracker, flags, and config. Use `.fit(Y)` to train.

2. **Probabilistic hierarchy**: Components organized by probabilistic level:
   - **Priors**: Hyperpriors and priors with `sample()` methods
   - **Posteriors**: Variational posterior estimates with `mean`, `cov`, `moment` attributes
   - **Realizations**: Concrete parameter values (samples or point estimates)

3. **Posterior access**: After fitting, access results via:
   - `model.obs_posterior` — observation model posterior (`ObsParamsPosterior`)
   - `model.latents_posterior` — latent variable posterior (`LatentsPosteriorStatic`)

4. **Observation data** (`ObsStatic`): Stores multi-group data as a stacked array with `dims` specifying group boundaries. Use `.get_groups()` to get list of per-group views.

5. **ArrayContainer base class**: Provides custom `__repr__` showing shapes instead of values, plus `copy()` and `clear()` methods.

### Fitting workflow

```python
from latents.gfa import GFAFitConfig, GFAModel
from latents.callbacks import ProgressCallback

# Configure fitting parameters (frozen dataclass)
config = GFAFitConfig(
    x_dim_init=10,   # Initial latent dimensionality
)

# Instantiate and fit (automatically initializes if needed)
model = GFAModel(config=config)
model.fit(Y, callbacks=[ProgressCallback()])  # Optional progress bar

# Access results
model.obs_posterior         # Observation model posterior
model.latents_posterior     # Latent variable posterior
model.tracker               # Lower bound and runtime per iteration
model.tracker.plot_lb()     # Plot convergence
model.flags                 # Convergence status
model.flags.display()       # Print status summary

# Infer latents for new data
X_new = model.infer_latents(Y_new)
```

Available callbacks:
- `ProgressCallback` — tqdm progress bar
- `LoggingCallback` — structured logging to "latents" logger
- `CheckpointCallback` — periodic model checkpoints (loadable via `GFAModel.load()`)

See the `examples/gfa/` directory for complete examples with simulation, fitting, and visualization.

## Code style

- Ruff for linting and formatting (configured in pyproject.toml)
- NumPy docstring convention
- Type hints with `from __future__ import annotations`
- Test directory mirrors `src/latents/`; `@pytest.mark.fit` separates slow fitting tests
- Array shape comments on complex operations

See the [contributing guide](docs/source/development/contributing.md) for full style and testing conventions.

## Git and GitHub

### Commits

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format:
- Types: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `deps:`, `perf:`, `ci:`
- Present tense, imperative mood ("Add feature" not "Added feature")
- First line ≤72 characters

```sh
# Pre-commit hooks require uv run
uv run git commit -m "feat: add cross-validation support to GFA"
```

### Issues

Use the appropriate template from `.github/ISSUE_TEMPLATE/` when creating issues (bug reports, feature requests, documentation improvements, etc.).

### Pull requests

When creating PRs, **always follow the template** in `.github/PULL_REQUEST_TEMPLATE.md`:

1. Fill out the Description section
2. Link related issues
3. Check the appropriate Type of Change
4. Complete all applicable Checklist items:
   - Tests pass: `uv run pytest`
   - Code checks pass: `uv run pre-commit run --all-files`
   - Docs build without warnings: `uv run sphinx-build -W -b html docs/source docs/_build/html`
   - Update `CHANGELOG.md` for user-facing changes (see below)

## CHANGELOG

Update `CHANGELOG.md` for user-facing changes. Add entries under `[Unreleased]` using
[Keep a Changelog](https://keepachangelog.com/) categories:

- **Added** — new features
- **Changed** — changes to existing functionality
- **Deprecated** — soon-to-be removed features
- **Removed** — removed features
- **Fixed** — bug fixes
- **Security** — vulnerability fixes

On release, the continuous delivery workflow extracts the version's section for GitHub
Release notes.

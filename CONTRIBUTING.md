# Contributing to Latents <!-- omit in toc -->

Thank you for contributing to the Latents toolkit! Your time and effort is deeply
appreciated!

### Table of Contents

- [Code of Conduct](#code-of-conduct)
- [New Contributors](#new-contributors)
- [Types of Contributions](#types-of-contributions)
- [How to Contribute](#how-to-contribute)
- [Environment Setup](#environment-setup)
- [Code Checks and Linting](#code-checks-and-linting)
- [Testing](#testing)
- [Documentation](#documentation)
- [Style Guides](#style-guides)

## Code of Conduct

Everyone participating in this project is expected to uphold the
[Latents Code of Conduct](./CODE_OF_CONDUCT.md). Please read it before getting
started.

## New Contributors

If you haven't already, check out the [README](./README.md) file for an overview
of the project.

New to collaborating and contributing with Git and GitHub? The
[GitHub Docs](https://docs.github.com/en) provide many
resources for getting acquainted with these tools. Here are a few entry points:

- [Set up Git](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

## Types of Contributions

There are many ways to meaningfully contribute to the Latents project. The types of
contributions we welcome include, but are not limited to, the following.

### Bug reports and fixes

Simply creating an Issue to point out a bug helps improve this project. If you already
have an idea on how to fix the bug, consider submitting a Pull Request - we might ask
you to submit one ourselves!

### Feature or method requests and enhancements

Enhancements can include API improvements, runtime improvements, or better support for
pre- and post-processing of data and model fits. At a larger scale, we might consider
additional statistical methods if they fit within the scope of this project.

### Documentation improvements

We strive to keep documentation comprehensive, clear, and correct. Reports of missing,
unclear, or incorrect documentation (including simple typos), and suggestions for
improvement are all welcome. You could even suggest improvements to this file!

### API and usage tutorials

We welcome suggested improvements to the existing
[examples](https://latents.readthedocs.io/en/latest/auto_examples/index.html).
Other cool usage examples, or tutorials on the statistical methods themselves are also
valuable contributions.

### Unit and integration tests

Any bug fixes or feature enhancements should come with appropriately modified or
created tests. But filling in tests where you think they might be missing is a valuable
contribution in its own right!

### Benchmarks

Benchmarks can include runtime or statistical performance. For example, how well
does each method in Latents recover ground truth parameters as a function of number
of training samples?

## How to Contribute

### Issues

We use [Issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue)
to track work that needs to be done. When you open an issue, consider using an
[Issue Template](.github/ISSUE_TEMPLATE). We have Issue Templates for each of the
[contribution types](#types-of-contributions) above.

### Pull requests

To add your contribution to the Latents project, create a
[Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/best-practices-for-pull-requests).
Check out our [pull request template](.github/PULL_REQUEST_TEMPLATE.md). Keep pull
requests and individual commits narrow in scope. Your pull request will need to be
reviewed and approved before it can be merged with the main branch. Each pull request
will also undergo several [automated status checks](.github/workflows) (see
[Code Checks and Linting](#code-checks-and-linting), [Testing](#testing), and
[Documentation](#documentation) below).

### AI-assisted review

External contributors receive automated code review from Claude on pull requests.
The review covers code quality, potential bugs, performance, security, and test
coverage. This provides fast initial feedback—a human maintainer will also review
your PR.

You can mention `@claude` in any issue or PR comment to ask questions or request
assistance.

### Changelog

We maintain a [CHANGELOG.md](./CHANGELOG.md) following
[Keep a Changelog](https://keepachangelog.com/) conventions. For user-facing changes,
add an entry under the `[Unreleased]` section:

- **Added** — new features
- **Changed** — changes to existing functionality
- **Deprecated** — soon-to-be removed features
- **Removed** — removed features
- **Fixed** — bug fixes
- **Security** — vulnerability fixes

The release workflow automatically extracts your entry for GitHub Release notes.

## Environment Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

To install uv, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Quick Start

```sh
# Clone the repository
git clone https://github.com/egokcen/latents.git
cd latents

# Install all dependencies (creates .venv automatically)
uv sync --all-groups

# Verify installation
uv run pytest
```

That's it! The `uv sync` command automatically creates a virtual environment in `.venv`
and installs all dependencies. You don't need to manually activate the environment—just
prefix commands with `uv run`.

### Dependency Groups

Dependencies are organized into groups in [pyproject.toml](./pyproject.toml):

- **dev**: Development tools (pre-commit, ruff)
- **test**: Testing tools (pytest, pytest-cov, coverage)
- **doc**: Documentation tools (sphinx, pydata-sphinx-theme, etc.)

To install only specific groups:

```sh
uv sync --group dev --group test  # Skip doc dependencies
```

## Code Checks and Linting

### pre-commit

We use [pre-commit](https://pre-commit.com/) for code checks and linting.
You can find the current set of pre-commit hooks [here](./.pre-commit-config.yaml).
To install the git hook scripts in your cloned repository, run

```sh
uv run pre-commit install
```

pre-commit will now run automatically on `git commit`. Because pre-commit needs
access to the tools installed in the virtual environment, use `uv run` when committing:

```sh
uv run git commit -m "your commit message"
```

You can also run pre-commit manually on all files:

```sh
uv run pre-commit run --all-files
```

### Ruff

Under the hood, we've configured pre-commit to use
[Ruff](https://docs.astral.sh/ruff/) for linting and formatting. See the
[pyproject.toml](./pyproject.toml) for the current ruleset. We highly recommend
configuring your development environment to automatically employ Ruff with these
rules (see, for example, the
[VS Code Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)).

You can also run Ruff linting and formatting explicitly via the command line:

```sh
uv run ruff check        # Run the linter
uv run ruff format       # Run the formatter
```

## Testing

### pytest

Testing in the Latents project relies on [pytest](https://docs.pytest.org/en/stable/):

- Tests live in the [`tests`](./tests) directory.
- The `tests` directory contains appropriately named subdirectories (`test_*`) for each
  subpackage or module belonging to the `latents` package.
- Test modules follow a similar naming convention (`test_*.py`).
- Each test function defined in a test module must have a name that starts with "test"
  (e.g., `test_fit`).

These naming conventions are critical for pytest to accurately discover and run all
tests in the project.

Run all tests in the project:

```sh
uv run pytest
```

Coverage is automatically calculated and reported (configured in pyproject.toml).

### Testing on Multiple Python Versions

To test on a specific Python version, re-sync the environment with that version:

```sh
# Test on Python 3.10
uv sync --python 3.10 --all-groups
uv run pytest

# Test on Python 3.14
uv sync --python 3.14 --all-groups
uv run pytest
```

uv will automatically download and manage Python versions as needed. See the
[uv Python version documentation](https://docs.astral.sh/uv/concepts/python-versions/)
for more details.

## Documentation

### Sphinx

We use [Sphinx](https://www.sphinx-doc.org/en/master/) to build our documentation.
As a developer, if you're mostly just working on code, then the [docstring conventions](#docstrings) are the most important information for you to keep in mind.
If you keep to the style guide, then Sphinx will do much of the heavy lifting based
on your docstrings.

Documentation source files live [here](./docs/source). The source directory structure
aims to match the hierarchical structure of the documentation site itself. Required
Sphinx extensions and core configurations are set in [conf.py](./docs/source/conf.py).
Source files are written in reStructuredText (`*.rst`) or Markdown (`*.md`).

To build the documentation:

```sh
uv run sphinx-build -W -b html docs/source docs/_build/html
```

To check for broken links in the documentation:

```sh
uv run sphinx-build -b linkcheck docs/source docs/_build/linkcheck
```

Locally serve the built HTML pages:

```sh
uv run python -m http.server -b 127.0.0.1 8000 -d docs/_build/html
```

`sphinx-autobuild` can be helpful to locally serve and automatically rebuild the
documentation upon revision:

```sh
uv run sphinx-autobuild docs/source docs/_build/html
```

## Style Guides

### Git Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

- Use commit types: `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`
- Use the present tense ("Add parameter" not "Added parameter")
- Use the imperative mood ("Infer latent..." not "Infers latent...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Examples:
```text
feat: add cross-validation support to GFA
fix: correct posterior covariance calculation
docs: update installation instructions for uv
chore: update pre-commit hooks
```

### Python Code

We use [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting. See
the [pyproject.toml](./pyproject.toml) for the current ruleset. Pull Requests will need
to meet these rules to be accepted.

### Array Shape Comments

For complex array operations, especially broadcasting, add inline shape comments:

```python
# Input shapes -> output shape
# X: (x_dim, n_samples), Y: (y_dim, n_samples) -> XY: (x_dim, y_dim)
XY = X @ Y.T

# Explain broadcasting expansions
# d: (y_dim,) -> (y_dim, 1) for broadcast with Y: (y_dim, n_samples)
centered = Y - d[:, np.newaxis]  # (y_dim, n_samples)
```

Use semantic dimension names:

| Dimension | Name | Description |
|-----------|------|-------------|
| Samples | `n_samples` | Independent observations |
| Time points | `n_time` | Time steps per sequence |
| Latent dims | `x_dim` | Latent space dimensionality |
| Observed dims | `y_dim` | Observation space dimensionality |
| Groups | `n_groups` | Number of observation groups |

### Docstrings

We use [NumPy style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html)
with [type hints](https://docs.python.org/3/library/typing.html). Sphinx builds documentation
automatically from docstrings, so following these conventions ensures proper rendering.

#### Required Sections by Object Type

| Object Type | Required Sections | Notes |
|-------------|-------------------|-------|
| Function | Parameters, Returns | Include Raises if exceptions are intentional |
| Class | Parameters, Attributes | Parameters = `__init__` args; Attributes = post-construction state |
| Dataclass | Parameters | Document fields in Parameters (not Attributes) |
| Property | One-line summary | Simple accessors need only a brief description |
| Method | Parameters, Returns | Same as functions |

#### Type and Default Formatting

```python
# Union types: use "or" not "|"
config : GFAFitConfig or None, default None

# Optional with default: include both
callbacks : list of Callback or None, default None

# Array shapes (for raw ndarray parameters)
data : ndarray of shape (y_dim, n_samples)
```

#### Returns Section

```python
# Single return: type only, no name
Returns
-------
LatentsPosteriorStatic
    Posterior over latent variables.

# Multiple returns: name each value
Returns
-------
obs_posterior : ObsParamsPosterior
    Fitted observation model posterior.
tracker : GFAFitTracker
    Quantities tracked during fitting.
```

#### Inline Code Formatting

Use backticks for inline code (variable names, attribute access, shapes, literals):

```python
# Variable names and attribute access
"""If `obs_posterior.y_dims` does not match `Y.dims`."""

# Array shapes
"""Array of shape `(n_groups, x_dim)`."""

# Parameter values and literals
"""Set `max_iter=1000` for longer runs."""
```

Both single and double backticks work (we set `default_role = "code"` in Sphinx).

#### Cross-References

Use Sphinx roles to create clickable links to documented objects. The format
depends on whether the name is unique or could be ambiguous across modules.

**Classes** — use short names, as long as they are unique:

```python
See :class:`GFAFitConfig` for options.
```

**Methods** — use `ClassName.method` format (class name disambiguates):

```python
Load a saved model with :meth:`GFAModel.load`.
```

**Module-level functions** — use full path with tilde (function names like `fit`,
`simulate`, `infer_latents` exist in multiple modules):

```python
# Tilde displays just "fit" but links to the correct function
See :func:`~latents.gfa.inference.fit` for details.
```

**Modules** — use full path with tilde:

```python
See :mod:`~latents.callbacks` for available callbacks.
```

**When to use cross-references vs. backticks:**

| Use case | Format |
|----------|--------|
| Documented class/function (clickable link) | `:class:`GFAModel``, `:func:`~latents.gfa.inference.fit`` |
| Variable names, attributes | ``` ``obs_posterior.y_dims`` ``` |
| Array shapes | ``` ``(n_groups, x_dim)`` ``` |
| Parameter values, literals | ``` ``max_iter=1000`` ``` |

#### Examples Section

Include Examples for:
- Public entry-point functions (``fit``, ``simulate``, ``infer_latents``)
- User-facing classes (``GFAModel``, ``GFAFitConfig``, ``ObsStatic``)

Use bold headers to organize multiple examples:

```python
Examples
--------
**Basic usage**

>>> model = GFAModel(config=GFAFitConfig(x_dim_init=10))
>>> model.fit(Y)

**With callbacks**

>>> from latents.callbacks import ProgressCallback
>>> model.fit(Y, callbacks=[ProgressCallback()])
```

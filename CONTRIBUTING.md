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

#### Bug reports and fixes

Simply creating an Issue to point out a bug helps improve this project. If you already
have an idea on how to fix the bug, consider submitting a Pull Request - we might ask
you to submit one ourselves!

#### Feature or method requests and enhancements

Enhancements can include API improvements, runtime improvements, or better support for
pre- and post-processing of data and model fits. At a larger scale, we might consider
additional statistical methods if they fit within the scope of this project.

#### Documentation improvements

We strive to keep documentation comprehensive, clear, and correct. Reports of missing,
unclear, or incorrect documentation (including simple typos), and suggestions for improvement are all welcome. You could even suggest improvements to this file!

#### API and usage tutorials

We welcome suggested improvements to the existing [tutorial notebooks](./notebooks).
Other cool usage examples, or tutorials on the statistical methods themselves are also
valuable contributions.

#### Unit and integration tests

Any bug fixes or feature enhancements should come with appropriately modified or
created tests. But filling in tests where you think they might be missing is a valuable
contribution in its own right!

#### Benchmarks

Benchmarks can include runtime or statistical performance. For example, how well
does each method in Latents recover ground truth parameters as a function of number
of training samples?

## How to Contribute

#### Issues

We use [Issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue)
to track work that needs to be done. When you open an issue, consider using an
[Issue Template](.github/ISSUE_TEMPLATE). We have Issue Templates for each of the 
[contribution types](#types-of-contributions) above.

#### Pull requests

To add your contribution to the Latents project, create a
[Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/best-practices-for-pull-requests).
Check out our [pull request template](.github/PULL_REQUEST_TEMPLATE.md). Keep pull
requests and individual commits narrow in scope. Your pull request will need to be
reviewed and approved before it can be merged with the main branch. Each pull request
will also undergo several [automated status checks](.github/workflows) (see
[Code Checks and Linting](#code-checks-and-linting), [Testing](#testing), and
[Documentation](#documentation) below).

## Environment Setup

#### Step 1: Clone the Latents repository

To work on Latents locally, clone the repository to your working directory:

```sh
git clone https://github.com/egokcen/latents.git
```

#### Step 2: Create a clean Python environment

Similar to the instructions laid out in the [README](./README.md),
if you are a 
[venv](https://docs.python.org/3/library/venv.html) user (preferred for developers),
run

```sh
python -m venv myenv
```

This command will create a new directory ``myenv`` (choose any name you like).
To activate the virtual environment, run

```sh
source myenv/bin/activate
```

All of your development should be done inside of this environment.

If you are a 
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 
user, run

```sh
conda create --name myenv python=3.10
conda activate myenv
```

Again, you can choose any name you like for the virtual environment, and any
[supported version](https://devguide.python.org/versions/) of Python.

#### Step 3: Install the latents Python package


You're now ready to install the `latents` Python package and its current dependencies.
Navigate to the repository directrory:

```sh
cd latents
```

Install the package in editable mode, with additional optional dependencies for
development and documentation:

```sh
pip install -e .[dev,doc]
```

These extra dependencies are laid out in the [pyproject.toml](./pyproject.toml) file.
If you don't need one of these sets of dependencies, you can omit `dev` or `doc`.

#### Step 4 (Optional): Install Nox

Optionally, install [Nox](https://nox.thea.codes/en/stable/):

```sh
pip install nox
```

Nox is helpful, in particular, for automating local [testing](#testing) across 
multiple Python versions ([pyenv](https://github.com/pyenv/pyenv) is a great tool for
switching between Python versions on your machine, and it plays nicely with Nox). In the
[noxfile.py](./noxfile.py) file, there also sessions configured for
[code checks and linting](#code-checks-and-linting) and building
[documentation](#documentation).

## Code Checks and Linting

#### pre-commit

We use [pre-commit](https://pre-commit.com/) for code checks and linting.
You can find the current set of pre-commit hooks [here](./.pre-commit-config.yaml).
To install the git hook scripts in your cloned repository, run

```sh
pre-commit install
```

pre-commit will now run automatically on `git commit`. It can also be run manually:

```sh
pre-commit run --all-files --show-diff-on-failure
```
#### Ruff

Under the hood, we've configured pre-commit to use 
[Ruff](https://docs.astral.sh/ruff/) for linting and formatting. See the
[pyproject.toml](./pyproject.toml) for the current ruleset. We highly recommend
configuring your development environment to automatically employ Ruff with these
rules (see, for example, the
[VS Code Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)).

You can also run Ruff linting and formatting explicitly via the command line.
To run the linter:

```sh
ruff check
```

If all checks pass, you can run the formatter:
```sh
ruff format
```

#### Nox

With pre-commit set up, you can optionally perform the above tasks with
[Nox](#step-4-optional-install-nox):

```sh
nox -s lint
```

## Testing

#### pytest

Testing in the Latents project relies on [pytest](https://docs.pytest.org/en/stable/):

- Tests live in the [`tests`](./tests) directory.
- The `tests` directory contains appropriately named subdirectories (`test_*`) for each
  subpackage or module belonging to the `latents` package.
- Test modules follow a similar naming convention (`test_*.py`).
- Each test function defined in a test module must have a name that starts with "test"
  (e.g., `test_fit`).

These naming conventions are critical for pytest to accurately discover and run all
tests in the project.

Run all tests in the project by invoking pytest in the command line:

```sh
pytest
```

To calculate coverage and generate a coverage report, run pytest with these optional
arguments:

```sh
pytest --cov=latents --cov-report=xml
```

#### Nox

You can run the project's tests across all supported versions of Python with the help
of Nox:

```sh
nox -s tests
```

For this session to run successfully, you will need to have all supported versions
of Python locally accessible on your machine. [pyenv](https://github.com/pyenv/pyenv)
is a great tool for that purpose.

## Documentation

#### Sphinx

We use [Sphinx](https://www.sphinx-doc.org/en/master/) to build our documentation.
As a developer, if you're mostly just working on code, then the documentation [style guide](#documentation-1) is the most important information for you to keep in mind.
If you keep to the style guide, then Sphinx will do much of the heavy lifting based
on your docstrings.

Documentation source files live [here](./docs/source). The source directory structure
aims to match the hierarchical structure of the documentation site itself. Required
Sphinx extensions and core configurations are set in [conf.py](./docs/source/conf.py).
Source files are written in reStructuredText (`*.rst`).

To build the documentation:

```sh
sphinx-build -M html docs/source docs/build
```

To check for broken links in the documentation:

```sh
sphinx-build -M linkcheck docs/source docs/build
```

Locally serve the built html pages:

```sh
python -m http.server -b 127.0.0.1 8000 -d docs/build/html
```

`sphinx-autobuild` can be helpful to locally serve and automatically rebuild the
documentation upon revision:

```sh
sphinx-autobuild docs/source/ docs/build/html/
```

#### Nox

You can use Nox to take care of building the documentation:

```sh
nox -s docs
```

To build and then serve the documentation locally:

```sh
nox -s docs -- --serve
```

## Style Guides

### Git Commit Messages

- Use the present tense ("Add parameter" not "Added parameter")
- Use the imperative mood ("Infer latent..." not "Infers latent...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### Python Code

We use [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting. See
the [pyproject.toml](./pyproject.toml) for the current ruleset. Pull Requests will need
to meet these rules to be accepted.

### Documentation

To facilitate [Sphinx's](https://www.sphinx-doc.org/en/master/) the automated build process, we stick to 
[NumPy style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html)
with [type hints](https://docs.python.org/3/library/typing.html).
Docstrings include instances of explicit markup syntax and directives 
(e.g., `**bold_variable**` or `:class:`) for proper rendering. Look through any
[source file](./src) for examples.
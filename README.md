# Latents

[![PyPI Version](https://img.shields.io/pypi/v/latents.svg)](https://pypi.python.org/pypi/latents)
[![License](https://img.shields.io/pypi/l/latents.svg)](https://pypi.python.org/pypi/latents)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Build Status](https://github.com/egokcen/latents/actions/workflows/pytest.yml/badge.svg)](https://github.com/egokcen/latents/actions/workflows/pytest.yml)

**Latents** is a toolkit for latent variable modeling and dimensionality reduction,
with an emphasis on linear, probabilistic methods.

> [!WARNING]
> The **Latents** toolkit is in an early stage of development and is not yet stable.
> The API is subject to frequent change.

## Table of Contents

- [Supported Methods](#supported-methods)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Code of Conduct](#code-of-conduct)
- [License](#license)

## Supported Methods
**Latents** currently supports the following methods:
- Group factor analysis (GFA)
- Delayed latents across (multiple) groups (mDLAG)

For information on each method, see the [documentation](#documentation).

## Getting Started

### Prerequisites

**Latents** is implemented in [Python](https://www.python.org/downloads/). It 
accomodates all [supported Python releases](https://devguide.python.org/versions/).
To proceed to [installation](#installation), you should have Python installed on your
machine.

[Usage demonstrations and tutorials](notebooks) for the ``latents`` package are provided in 
[Jupyter Notebooks](https://jupyter.org/). Jupyter Notebook dependencies will be
installed when you install the ``latents`` package, but familiarity with notebooks
will be helpful to take full advantage of these resources.

### Installation

Assuming you have Python installed on your machine, start by creating a virtual 
environment inside your project directory. For example, if you are a 
[venv](https://docs.python.org/3/library/venv.html) user, run

```sh
python -m venv myenv
```

This command will create a new directory ``myenv`` (choose any name you like).
To activate the virtual environment, run

```sh
source myenv/bin/activate
```

If you are a 
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 
user, run

```sh
conda create --name myenv python=3.10
conda activate myenv
```

Again, you can choose any name you like for the virtual environment, and any
supported version of Python.

Next, navigate to the ``latents`` directory. Install the package locally using the
following command:

```sh
python3 -m pip install .
```

Alternatively, you can install the package in editable mode, which allows you
to modify the source code and have the changes reflected immediately, without
having to reinstall the package. To do so, run

```sh
python3 -m pip install -e .
```

> [!NOTE]
> Depending on your system, you may need to explicitly provide the path to
> the ``latents`` package. In that case, replace ``.`` with the path to the ``latents``
> directory.    

You can now import and use the ``latents`` package wherever your project is located.

## Documentation
The official documentation is hosted on **Read the Docs**.

## Contributing

Interested in contributing? See the [Contributing Guide](.github/CONTRIBUTING.md).

## Code of Conduct

Whether you are a user, contributor, or maintainer, please consult the 
[Code of Conduct](.github/CODE_OF_CONDUCT.md).

## License
[MIT](LICENSE)

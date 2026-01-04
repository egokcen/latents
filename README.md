# Latents

[![PyPI Version](https://img.shields.io/pypi/v/latents.svg)](https://pypi.org/project/latents/)
[![Python Versions](https://img.shields.io/pypi/pyversions/latents.svg)](https://pypi.org/project/latents/)
[![License](https://img.shields.io/pypi/l/latents.svg)](https://pypi.org/project/latents/)

[![CI](https://github.com/egokcen/latents/actions/workflows/ci.yml/badge.svg)](https://github.com/egokcen/latents/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/egokcen/latents/branch/main/graph/badge.svg)](https://codecov.io/gh/egokcen/latents)
[![Documentation](https://readthedocs.org/projects/latents/badge/?version=latest)](https://latents.readthedocs.io)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

**Latents** is a Python toolkit for latent variable modeling and dimensionality reduction,
with an emphasis on linear, probabilistic methods.

> [!WARNING]
> The **Latents** toolkit is in an early stage of development and is not yet stable.
> The API is subject to frequent change.

## Supported Methods

- **Group Factor Analysis (GFA)**
- **Delayed Latents Across Multiple Groups (mDLAG)** — under development

## Installation

Requires Python 3.10 or higher.

```sh
pip install latents
```

For Jupyter notebook support:

```sh
pip install latents[notebook]
```

## Quick Start

```python
from latents.gfa import GFAFitConfig, GFAModel

# Configure fitting parameters
config = GFAFitConfig(
    x_dim_init=10,  # Initial latent dimensionality
    verbose=True,
)

# Instantiate and fit to multi-group observation data Y
model = GFAModel(config=config)
model.fit(Y)

# Check convergence and access results
model.flags.display()       # Fitting status
model.obs_posterior         # Observation model posterior
model.latents_posterior     # Latent variable posterior
model.tracker.plot_lb()     # Plot lower bound convergence
```

See the [examples gallery](https://latents.readthedocs.io/en/latest/auto_examples/index.html)
for complete tutorials.

## Documentation

Full documentation, including tutorials and API reference, is available at
[latents.readthedocs.io](https://latents.readthedocs.io).

## Contributing

Interested in contributing? See the [Contributing Guide](CONTRIBUTING.md).

## Code of Conduct

Please consult the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE)

<h1 align="center">
  <a href="https://latents.readthedocs.io">
    <img src="docs/source/_static/latents-github-banner.svg" alt="Latents logo">
  </a>
</h1>

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/latents.svg)](https://pypi.org/project/latents/)
[![Python Versions](https://img.shields.io/pypi/pyversions/latents.svg)](https://pypi.org/project/latents/)
[![License](https://img.shields.io/pypi/l/latents.svg)](LICENSE)

[![CI](https://github.com/egokcen/latents/actions/workflows/ci.yml/badge.svg)](https://github.com/egokcen/latents/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/egokcen/latents/branch/main/graph/badge.svg)](https://codecov.io/gh/egokcen/latents)
[![Documentation](https://readthedocs.org/projects/latents/badge/?version=latest)](https://latents.readthedocs.io)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

</div>

## Latents: A Python Library for Latent Variable Modeling

> [!WARNING]
> Latents is in an early stage of development and is not yet stable.
> The API is subject to frequent change.

Latents is a Python library for latent variable modeling and dimensionality reduction,
with an emphasis on linear, probabilistic methods. The following methods are currently
supported:

- Group Factor Analysis (GFA)
- Delayed Latents Across Multiple Groups (mDLAG) — under development

## Installation

Requires Python 3.10 or higher.

```sh
pip install latents
```

## Quick Start

```python
from latents.callbacks import ProgressCallback
from latents.gfa import GFAFitConfig, GFAModel

# Configure fitting parameters
config = GFAFitConfig(x_dim_init=10)  # Initial latent dimensionality

# Instantiate and fit to multi-group observation data Y
model = GFAModel(config=config)
model.fit(Y, callbacks=[ProgressCallback()])  # Progress bar during fitting

# Check convergence and access results
model.flags.display()       # Fitting status
model.obs_posterior         # Observation model posterior
model.latents_posterior     # Latent variable posterior
model.tracker.plot_lb()     # Plot lower bound convergence
```

## Documentation

See our [documentation](https://latents.readthedocs.io) for the complete API reference,
user guide, and tutorials.

## Contributing

Interested in contributing? See the [Contributing Guide](CONTRIBUTING.md).

## Code of Conduct

Please consult the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE)

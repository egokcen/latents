(methods)=

# Methods

Latents implements methods from a family of probabilistic generative models.
Each method defines a **state model** (distribution over latents) and an
**observation model** (how observations arise from latents).

## Method family

|                    | 1 Group   | 2 Groups | N Groups |
|--------------------|-----------|----------|----------|
| **Static**         | pPCA → FA | pCCA     | GFA      |
| **GP time series** | GPFA      | DLAG     | mDLAG    |

- **Rows** share a state model (static or Gaussian process time series)
- **Columns** share observation model structure (number of observed groups)
- **pPCA → FA** indicates FA generalizes pPCA (anisotropic vs isotropic noise)
- **GFA** and **mDLAG** use automatic relevance determination (ARD) for
  automatic dimensionality selection

:::{tip}
See {doc}`../foundations/mathematical_background` for the shared generative
framework underlying all methods.
:::

## Choosing a method

**How many observation groups do you have?**
: One group → single-group methods (left column). Multiple groups with shared
  latent structure → multi-group methods (right columns).

**Is your data static or time series?**
: Independent samples → static methods (top row). Temporal dependencies →
  time series methods (bottom row).

**Do you need automatic dimensionality selection?**
: If yes, use Bayesian methods with ARD: **GFA** (static) or **mDLAG** (time series).

## Available methods

:::{note}
Latents currently implements **GFA** (complete) and **mDLAG** (under development).
:::

```{toctree}
:maxdepth: 1

gfa
mdlag
```

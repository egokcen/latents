(reproducibility)=

# Reproducibility

Probabilistic models involve random sampling and random initialization.
Reproducing results requires controlling these sources of randomness. Latents
uses NumPy's modern `Generator` API throughout, and every config class accepts
a `random_seed` parameter.

## Setting a random seed

Pass `random_seed` to any config to make results reproducible. The default
`None` draws fresh entropy each run.

```python
from latents.gfa import GFAModel, GFAFitConfig, GFASimConfig

# Reproducible simulation
sim_config = GFASimConfig(
    n_samples=100,
    y_dims=np.array([10, 10]),
    x_dim=5,
    random_seed=42,
)

# Reproducible fitting
fit_config = GFAFitConfig(x_dim_init=10, random_seed=42)
model = GFAModel(config=fit_config)
model.fit(Y)

# Running again with the same seed produces the same result
model2 = GFAModel(config=fit_config)
model2.fit(Y)
```

## What the seed controls

The seed controls different things depending on the context:

- **Simulation** (`simulate()`): sampling loading matrices from the prior,
  sampling latent variables, and generating observation noise. The same seed
  produces identical synthetic data.

- **Fitting** (`model.fit()`): initializing the posterior estimates. The
  variational EM updates that follow are deterministic given the starting
  point. The seed determines which local optimum the optimizer finds.

Because fitting uses the seed only at initialization, two runs with the same
seed and same data will converge to the same solution.

## Structured seeding for parallel experiments

When running a sweep of independent fits, each run needs its own seed.
Manually choosing unrelated integers works but is fragile. A better approach is
to pass a **sequence** of integers:

```python
base_seed = 42

for run_idx in range(10):
    config = GFAFitConfig(
        x_dim_init=10,
        random_seed=[run_idx, base_seed],
    )
    model = GFAModel(config=config)
    model.fit(Y)
```

This works because NumPy's
[`SeedSequence`](https://numpy.org/doc/stable/reference/random/parallel.html)
hashes integer sequences into statistically independent bit generators.
Sequences that differ in any element &mdash; such as `[0, 42]` and `[1, 42]`
&mdash; produce unrelated random streams. You can add as many index dimensions
as your experiment needs (e.g., `[sweep_idx, run_idx, base_seed]`).

## Further reading

- [NumPy parallel random number generation](https://numpy.org/doc/stable/reference/random/parallel.html)
  &mdash; how `SeedSequence` enables reproducible parallelism
- {doc}`/development/benchmarks` &mdash; the benchmark suite's three-stream
  seed design for separating data, noise, and fitting randomness
- {doc}`../methods/gfa` &mdash; GFA model specification and inference details

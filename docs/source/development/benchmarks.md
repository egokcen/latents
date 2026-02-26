(benchmarks)=

# Benchmarks

The `benchmarks/` directory contains performance and correctness benchmarks for
Latents methods. This page is for developers who want to run benchmarks, update
benchmark results in the documentation, or extend benchmarks to new methods.

For benchmark results, see the {doc}`../auto_benchmarks/index`.

## Purpose

Benchmarks serve two goals:

1. **User guidance** — Help users understand computational scaling and data
   requirements for each method
2. **Correctness verification** — Transparent proof that implementations recover
   ground truth parameters across a range of conditions, and meet the theoretical
   expectations for runtime scaling

## Directory structure

```text
benchmarks/
├── __init__.py
├── cli.py               # Main CLI entry point
├── metrics.py           # subspace_error, relative_l2_error, denoised_r2
├── plotting.py          # Shared plotting utilities for gallery scripts
├── system_info.py       # Hardware/software info collection
│
├── gallery/             # Sphinx-gallery scripts and committed data
│   ├── README.rst
│   ├── data/            # Pre-computed results (committed)
│   │   ├── system_info.json
│   │   └── gfa/
│   │       ├── runtime/
│   │       │   ├── n_samples.csv
│   │       │   ├── x_dim.csv
│   │       │   └── ...
│   │       ├── recovery/
│   │       │   ├── n_samples.csv
│   │       │   ├── snr.csv
│   │       │   └── ...
│   │       └── dimensionality/
│   │           └── n_samples_snr.csv
│   └── gfa/             # GFA gallery scripts
│       ├── README.rst
│       ├── plot_1_runtime.py
│       ├── plot_2_recovery.py
│       └── plot_3_dimensionality.py
│
└── gfa/                 # GFA benchmark implementation
    ├── __init__.py
    ├── cli.py           # GFA CLI subcommands
    ├── config.py        # BenchmarkConfig, DimensionalityConfig, presets
    ├── data.py          # Ground truth generation, subsetting, SNR resampling
    ├── runtime.py       # Runtime sweep orchestration
    ├── recovery.py      # Parameter recovery sweep orchestration
    └── dimensionality.py  # Dimensionality recovery orchestration
```

The separation between `benchmarks/gfa/` (benchmark logic) and
`benchmarks/gallery/gfa/` (visualization scripts) keeps concerns separate:
benchmark code focuses on running experiments, while gallery scripts focus on
presenting results.

## Design principles

### One-factor-at-a-time sweeps

Runtime and recovery benchmarks vary one factor (e.g., sample size) while
holding all others at fixed defaults. This isolates each factor's effect,
producing interpretable scaling curves.

Default values differ between benchmark types. See `RUNTIME_CONFIG` and
`RECOVERY_CONFIG` in `benchmarks/gfa/config.py` for exact values.

### Full factorial design

The dimensionality benchmark uses a different approach: a full factorial grid
over `n_samples` × `snr`. This shows how both factors jointly affect
dimensionality selection. See `DIMENSIONALITY_CONFIG` in
`benchmarks/gfa/config.py`.

### Subsetting for controlled comparisons

For parameter recovery benchmarks, we want to isolate the effect of *available*
data on recovery quality while holding the underlying ground truth fixed. This
emulates realistic scenarios. In neuroscience, for example, experimenters may
have limited trials or can only record from a subset of neurons.

| Sweep | Strategy |
|-------|----------|
| n_samples | Generate once at max N, subset `Y[:, :n]` for smaller N |
| y_dim | Generate once at max D, subset dimensions for smaller D |
| x_dim | Fresh generation required (different latent structure) |
| n_groups | Fresh generation required (different group structure) |
| snr | Generate base params once, adjust noise via `resample_at_snr()` |

For `n_samples` and `y_dim`, subsetting ensures that differences in recovery
are due solely to the amount of available data, not to variation in the
underlying signal. Subsetting also avoids redundant data generation. For `x_dim`
and `n_groups`, the underlying data structure changes, so each sweep value
requires fresh generation.

The dimensionality benchmark combines these strategies: ground truth is
generated once per run at max `n_samples`, observations are resampled per SNR
level, then subsetted per sample size.

### Aggregation

Each sweep point runs 10 independent times with different random seeds. The
CSV files report mean and SEM (standard error of the mean) only — no per-run
data — to keep files small while providing uncertainty estimates.

## Running benchmarks

### Prerequisites

Benchmarks require the `benchmark` dependency group:

```sh
uv sync --group benchmark
```

Or sync all groups if you're also building docs:

```sh
uv sync --all-groups
```

### CLI structure

The CLI follows a `method → command → options` hierarchy:

```sh
python -m benchmarks.cli <method> <command> [options]
```

**Runtime benchmarks** (one-factor-at-a-time):

```sh
# Run all runtime sweeps (n_samples, y_dim, x_dim, n_groups)
python -m benchmarks.cli gfa runtime --all --workers 4

# Run a single sweep
python -m benchmarks.cli gfa runtime --sweep n_samples
```

**Recovery benchmarks** (one-factor-at-a-time):

```sh
# Run all recovery sweeps (n_samples, y_dim, x_dim, n_groups, snr)
python -m benchmarks.cli gfa recovery --all --workers 4

# Run a single sweep
python -m benchmarks.cli gfa recovery --sweep snr
```

**Dimensionality benchmarks** (full n_samples × snr grid):

```sh
python -m benchmarks.cli gfa dimensionality --workers 4
```

**Run everything:**

```sh
# All GFA benchmarks (runtime + recovery + dimensionality)
python -m benchmarks.cli gfa all --workers 4

# All methods (currently GFA only)
python -m benchmarks.cli all --workers 4
```

### Output structure

By default, results go to `benchmarks/results/` (gitignored):

```text
benchmarks/results/
├── system_info.json      # Hardware/software snapshot
└── gfa/
    ├── runtime/
    │   ├── n_samples.csv
    │   ├── x_dim.csv
    │   └── ...
    ├── recovery/
    │   ├── n_samples.csv
    │   ├── snr.csv
    │   └── ...
    └── dimensionality/
        └── n_samples_snr.csv
```

Each CSV contains aggregated results (mean ± SEM) across all runs for that
sweep.

### Updating documentation

The benchmark gallery displays pre-computed results from `benchmarks/gallery/data/`.
To update these results after running new benchmarks:

1. Run the benchmarks
2. Review the results for correctness
3. Copy finalized data to the gallery:

```sh
cp benchmarks/results/system_info.json benchmarks/gallery/data/
cp benchmarks/results/gfa/runtime/*.csv benchmarks/gallery/data/gfa/runtime/
cp benchmarks/results/gfa/recovery/*.csv benchmarks/gallery/data/gfa/recovery/
cp benchmarks/results/gfa/dimensionality/*.csv benchmarks/gallery/data/gfa/dimensionality/
```

4. Rebuild docs to verify plots render correctly
5. Commit the updated data files

## Extending benchmarks

### Adding a new method

To add benchmarks for a new method (e.g., FA):

1. **Create the method subdirectory** — `benchmarks/fa/` mirroring
   `benchmarks/gfa/`:

   - `config.py` — Define `BenchmarkConfig` with appropriate sweep ranges
   - `data.py` — Implement ground truth generation using the method's simulation
   - `runtime.py` — Runtime sweep orchestration
   - `recovery.py` — Parameter recovery sweep orchestration
   - `dimensionality.py` — Dimensionality recovery orchestration
   - `cli.py` — CLI subcommands for the method

2. **Register with the main CLI** — In `benchmarks/cli.py`, import and register
   the new method's CLI module

3. **Create gallery scripts** — Add `benchmarks/gallery/fa/` with:
   - `README.rst` describing the method's benchmarks
   - `plot_1_runtime.py`, `plot_2_recovery.py`, `plot_3_dimensionality.py`

4. **Update sphinx-gallery config** — Add the new gallery path to
   `subsection_order` in `docs/source/conf.py`

### Adding a new sweep

To add a new sweep dimension to an existing method:

1. **Configure the sweep** — Add a `SweepConfig` entry in `config.py` with
   default value and sweep values
2. **Implement sweep logic** — Update the relevant orchestration module to handle
   the new sweep, including any special subsetting or generation logic
3. **Add CLI support** — Register the sweep name in the CLI choices
4. **Update visualization** — Add the sweep to the plotting script

### Adding a new metric

To add a new benchmark metric (e.g., a new recovery measure):

1. **Implement the metric** — Add a function to `benchmarks/metrics.py`
2. **Collect the metric** — Update the sweep orchestration to compute and
   record the metric
3. **Visualize the metric** — Update plotting scripts to display the new metric

### Plotting utilities

Shared plotting functions in `benchmarks/plotting.py` ensure consistent
visualization across methods:

- `plot_runtime_sweep(df, title, xlabel, ref_slope, ref_label)` — 1×3 subplot
  figure for runtime benchmarks (time/iter, iterations, total time) with
  log-log scaling, shaded SEM bands, and optional reference lines

- `plot_recovery_sweep(df, title, xlabel, log_x)` — single panel with all
  recovery metrics overlaid (subspace error, L2 errors, denoised R²)

- `plot_dimensionality(df, title)` — single panel with n_samples on x-axis
  and separate curves per SNR level

- `display_system_info(path)` — prints formatted hardware/software information
  for the benchmark environment appendix

When adding a new method, reuse these utilities to maintain visual consistency
across the documentation.

## Reproducibility

### System information

Benchmarks record hardware and software versions in `system_info.json`. This
context is important because runtime benchmarks are sensitive to CPU, memory,
and library versions:

```json
{
  "timestamp": "2026-02-02T06:37:18+00:00",
  "hardware": {
    "cpu": "AMD Ryzen 9 6900HX with Radeon Graphics",
    "cpu_cores_physical": 8,
    "cpu_cores_logical": 16,
    "ram_gb": 60.6
  },
  "software": {
    "python": "3.11.14",
    "numpy": "2.3.5",
    "scipy": "1.16.3",
    "latents": "0.0.5.dev24+g86fb0ace6.d20260201"
  },
  "os": {
    "system": "Linux",
    "release": "6.17.9-76061709-generic",
    "distro": "Pop!_OS 22.04 LTS"
  }
}
```

This information is displayed at the bottom of each benchmark gallery page.

### Random seeds

Benchmarks need multiple independent random streams. These must be independent
(no correlation) yet reproducible (same seed → same results).

We use NumPy's `SeedSequence`, which hashes a list of integers into independent
streams. Each benchmark run constructs a structured seed:

```python
seed = [sweep_idx, run_idx, sweep_value_idx, stream_id, base_seed]
rng = np.random.default_rng(seed)
```

#### Why three streams?

A single benchmark run involves three distinct sources of randomness, each
requiring an independent stream:

| Stream | Purpose | Example |
|--------|---------|---------|
| `_STREAM_DATA` | Ground truth parameters and latents | Loading matrices C, latents X |
| `_STREAM_OBS` | Observation noise realization | ε in Y = CX + d + ε |
| `_STREAM_FIT` | Model fitting initialization | Initial parameter guesses |

Separating these streams allows controlled experiments. For example, to test
whether fitting is sensitive to initialization, you can vary `_STREAM_FIT` while
holding `_STREAM_DATA` and `_STREAM_OBS` fixed — same data, different starting
points. Similarly, varying `_STREAM_OBS` tests sensitivity to noise realizations
while keeping ground truth parameters constant.

#### Seeding by sweep type

How data seeds are constructed depends on whether the sweep uses subsetting
(where data structure stays fixed) or requires fresh generation (where data
structure changes):

**Subsetting sweeps** (`n_samples`, `y_dim`): The data seed depends only on the
run index, not the sweep value. This ensures all sweep values within a run share
the same ground truth — smaller values are strict subsets of larger ones.

```python
# Same data seed for all n_samples values within run 0
get_data_seed("n_samples", run_idx=0)  # No sweep_value_idx
```

**Structural sweeps** (`x_dim`, `n_groups`): The data seed includes the sweep
value index because each value requires fundamentally different ground truth
(different latent dimensionality or group structure).

```python
# Different data seed for x_dim=3 vs x_dim=5 within the same run
get_data_seed("x_dim", run_idx=0, sweep_value_idx=0)  # x_dim=3
get_data_seed("x_dim", run_idx=0, sweep_value_idx=1)  # x_dim=5
```

Observation and fitting seeds always include `sweep_value_idx` since each sweep
point needs its own noise realization and fitting initialization.

#### Using seed methods

For one-factor-at-a-time benchmarks, use `BenchmarkConfig` seed methods rather
than constructing seeds manually:

- `get_data_seed(sweep_name, run_idx, sweep_value_idx?)` — for generating
  ground truth parameters and data
- `get_obs_seed(sweep_name, run_idx, sweep_value_idx)` — for sampling
  observation noise
- `get_fit_seed(sweep_name, run_idx, sweep_value_idx)` — for model fitting

`DimensionalityConfig` provides analogous seed methods adapted for the 2D
grid: `get_data_seed(run_idx)`, `get_obs_seed(run_idx, snr_idx)`,
`get_fit_seed(run_idx, snr_idx, n_samples_idx)`.

These methods ensure consistent seed structure across all benchmarks,
maintaining reproducibility as the benchmark suite grows.

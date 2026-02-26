Group Factor Analysis (GFA)
---------------------------

Benchmarks for Group Factor Analysis covering runtime scaling and parameter
recovery.

**Runtime benchmarks** validate theoretical complexity:

- Linear scaling in samples (N) and observed dimensions (D)
- Cubic scaling in latent dimensions (K), though optimized LAPACK routines
  yield quadratic scaling at typical sizes

**Recovery benchmarks** verify parameter estimation across:

- Sample sizes and noise levels
- Observed and latent dimensionalities
- Number of groups

**Dimensionality recovery benchmarks** evaluate post-hoc latent dimension
selection across sample sizes and noise levels.

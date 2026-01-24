# Documentation Terminology

Canonical terms and their usage across the Latents documentation.

## Project Terms

| Canonical Term | Avoid | Notes |
|----------------|-------|-------|
| Latents | latents, LATENTS | The project's name is capitalized; the technical term can be lowercase in prose |
| GFA | gfa, Gfa | Uppercase acronym |
| mDLAG | MDLAG, mdlag | Mixed case as shown |
| Group Factor Analysis | group factor analysis | Title case for full name |

## Technical Terms

| Canonical Term | Avoid | Definition |
|----------------|-------|------------|
| posterior | posterior distribution | Shorthand acceptable in context |
| prior | prior distribution | Shorthand acceptable in context |
| latent variable | latent, hidden variable | Full term preferred on first use |
| observation model | emission model | Consistent with codebase |
| state model | latent model | Consistent with codebase |
| loading matrix | factor loadings, weight matrix | Consistent with codebase |
| ELBO | evidence lower bound | Acronym preferred after first use |
| ARD | automatic relevance determination | Acronym preferred after first use |

## Variable Names in Docs

These appear frequently in docstrings and examples:

| Variable | Meaning | Type/Shape |
|----------|---------|------------|
| `x_dim` | Latent dimensionality | int |
| `y_dim` | Observed dimensionality summed across groups | int |
| `y_dims` | Observed dimensionalities per group | list of int |
| `n_groups` | Number of observation groups | int |
| `n_samples` | Number of samples | int |
| `Y` | Observation data | ObsStatic or ndarray |
| `X` | Latent variables | ndarray of shape (x_dim, n_samples) |
| `C` | Loading matrix | ndarray of shape (y_dim, x_dim) |

## Code Object References

When referencing code objects in documentation:

| Object Type | How to Reference | Example |
|-------------|------------------|---------|
| Class | `:class:` role | `:class:`GFAModel`` |
| Method | `:meth:` role | `:meth:`GFAModel.fit`` |
| Function | `:func:` with full path | `:func:`~latents.gfa.inference.fit`` |
| Module | `:mod:` with full path | `:mod:`~latents.callbacks`` |
| Parameter | backticks | `` `x_dim` `` |
| Attribute | backticks | `` `model.obs_posterior` `` |

## Abbreviations

Define on first use, then use abbreviation:

| Abbreviation | Full Form | First Use Example |
|--------------|-----------|-------------------|
| GFA | Group Factor Analysis | "Group Factor Analysis (GFA) is..." |
| mDLAG | Delayed Latents Across Multiple Groups | "Delayed Latents Across Multiple Groups (mDLAG)..." |
| ELBO | Evidence Lower Bound | "...maximizes the Evidence Lower Bound (ELBO)..." |
| ARD | Automatic Relevance Determination | "...uses Automatic Relevance Determination (ARD)..." |
| API | Application Programming Interface | Generally known; no need to expand |
| CI/CD | Continuous Integration/Deployment | Generally known; no need to expand |

## Units and Formatting

| Item | Format | Example |
|------|--------|---------|
| Dimensions | lowercase with underscore | `x_dim`, `y_dim` |
| Counts | `n_` prefix | `n_groups`, `n_samples` |
| Shapes | parentheses | `(x_dim, n_samples)` |
| Ranges | en-dash | "iterations 1–100" |
| Code literals | backticks | `` `None` ``, `` `True` `` |

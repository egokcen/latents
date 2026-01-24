# Docstring Audit Conventions

Audit-specific guidance for checking Python docstrings. For the canonical style conventions, see the [contributing guide](../../../../docs/source/development/contributing.md#docstrings).

## Check Codes

See [check-codes.md](check-codes.md) for the full list of DOC-* codes.

## What "Public" Means

**Public** = not prefixed with underscore, or explicitly listed in `__all__`.

- `def fit(...)` — public
- `def _validate(...)` — private, skip
- `class GFAModel` — public
- `class _InternalHelper` — private, skip

## Systematic Checking Procedures

### 1. Signature-Docstring Consistency

For each function/method:

1. Extract parameters from signature (names, types, defaults)
2. Extract Parameters section from docstring
3. Compare:
   - Same parameters? (DOC-C001, DOC-C002)
   - Same order? (DOC-C003)
   - Compatible types? (DOC-C004, DOC-C005)

**Type translation rules:**
| Signature | Docstring |
|-----------|-----------|
| `int \| None` | `int or None` |
| `list[Callback]` | `list of Callback` |
| `dict[str, Any]` | `dict of str to Any` |
| `tuple[int, ...]` | `tuple of int` |

### 2. Inline Code Formatting

**Backtick style consistency** (DOC-F001):
1. Count single vs double backticks per file
2. Flag deviations from dominant style

**Unformatted code terms** (DOC-F002):
Search docstrings for patterns that should have backticks:

```
# Snake_case identifiers in prose
\b[a-z]+_[a-z_]+\b

# Method calls
\b[a-z_]+\(\)

# Attribute access
\b\w+\.\w+\b

# Common parameters
\b(x_dim|y_dim|n_groups|n_obs|max_iter)\b
```

Exclude matches already inside backticks.

### 3. Cross-Reference Validation

**Build public API list first:**
```
# Classes
^class [A-Z]\w+

# Functions (non-private)
^def [a-z][a-z0-9_]*\(
```

**Check reference formats:**

| Reference Type | Correct Format | Wrong Formats |
|----------------|----------------|---------------|
| Class | `:class:`GFAModel`` | `` `GFAModel` ``, `GFAModel` |
| Method | `:meth:`GFAModel.load`` | `:meth:`load`` |
| Function | `:func:`~latents.gfa.inference.fit`` | `:func:`fit``, `:func:`latents.gfa.inference.fit`` |
| Module | `:mod:`~latents.callbacks`` | `:mod:`callbacks`` |

**Regex for potential issues:**
```
# Backticked class names (should use :class:)
`[A-Z][a-zA-Z]+`

# Function ref without path
:func:`[a-z_]+`

# Plain CamelCase in prose
\b[A-Z][a-z]+[A-Z][a-z]+\b
```

### 4. Required Sections Check

| Object Type | Required | Check |
|-------------|----------|-------|
| Function with args | Parameters | Has `Parameters\n---` |
| Function with return | Returns | Has `Returns\n---` |
| Class with `__init__` args | Parameters | Document in class or `__init__` |
| Class | Attributes | Has `Attributes\n---` |
| Dataclass | Parameters | Fields go here, not Attributes |

**Dataclass detection:**
```python
@dataclass
class Foo:
    ...
```

### 5. Returns Section Format

**Single return** — type only, no name:
```
Returns
-------
LatentsPosteriorStatic
    Description.
```

**Multiple returns** — name each:
```
Returns
-------
obs_posterior : ObsParamsPosterior
    Description.
tracker : GFAFitTracker
    Description.
```

Detection: Count items under Returns. If >1 and any lacks `:`, flag DOC-S002.

## Pragmatic Guidelines

- Private methods (`_name`) don't need full docstrings — a one-liner is fine
- Simple one-line functions may only need a one-line docstring
- Properties only need a one-line summary
- Focus on public API completeness
- Use judgment on edge cases

## Domain Context

Common domain terms (acceptable in docstrings):
- **Posterior / Prior**: Bayesian inference terms
- **ELBO**: Evidence lower bound
- **Loading matrix**: Maps latents to observations
- **ARD**: Automatic Relevance Determination

Variable naming follows statistical conventions:
- `mu`, `sigma`, `alpha`, `beta` — standard notation
- `x_dim`, `y_dim` — latent and observed dimensions
- `n_groups`, `n_samples` — counts

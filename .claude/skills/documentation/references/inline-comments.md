# Inline Comment Conventions

Conventions for inline comments in source code (`src/**/*.py`).

See the [contributing guide](../../../../docs/source/development/contributing.md#style-guides) for general code style.

## Shape Comments

For complex array operations involving broadcasting or reshaping, add shape comments
showing the transformation:

```python
# Broadcast precision across groups
# psi: (n_groups,) -> (n_groups, 1, 1)
psi_expanded = psi[:, None, None]
```

### When to Add Shape Comments

- Broadcasting operations that change rank
- Reshaping operations
- Einsum or tensordot operations
- Any array manipulation where the resulting shape isn't obvious

### Format

```python
# variable: (input_shape) -> (output_shape)
```

Or for operations on multiple arrays:

```python
# A: (m, k), B: (k, n) -> C: (m, n)
```

## Section Comments

For long modules (>200 lines), use section headers to organize code:

```python
# =============================================================================
# Public API
# =============================================================================

def fit(...):
    ...

# =============================================================================
# Internal helpers
# =============================================================================

def _validate_inputs(...):
    ...
```

### Standard Section Names

| Section | Contents |
|---------|----------|
| Public API | User-facing functions and classes |
| Internal helpers | Private functions (prefixed with `_`) |
| Constants | Module-level constants |
| Type definitions | TypeVar, Protocol, type aliases |

Section names can also semantically reflect the section's purpose.
For example: "Inference functions for individual parameters" or
"Lower bound computation".

## TODO/FIXME/NOTE Comments

Use these markers consistently:

| Marker | Purpose | Example |
|--------|---------|---------|
| `TODO` | Planned improvements, not blocking | `# TODO: Add support for missing data` |
| `FIXME` | Known issues that should be addressed | `# FIXME: Assumes positive-definite input` |
| `NOTE` | Important context for understanding | `# NOTE: Order matters for numerical stability` |

### Format

Always include a brief description:

```python
# TODO: Add support for missing data (issue #42)
# FIXME: This assumes positive-definite input; add validation
# NOTE: Order matters here due to numerical stability
```

### Linking to Issues

When a TODO or FIXME relates to a GitHub issue, include the issue number:

```python
# TODO: Implement batch processing (issue #123)
```

## Magic Numbers

Avoid unexplained numeric literals. Either:

1. Define as a named constant with a docstring
2. Add an inline comment explaining the value

**Bad**:
```python
if iterations > 1000:
    ...
```

**Good**:
```python
MAX_ITERATIONS = 1000  # Default convergence limit
if iterations > MAX_ITERATIONS:
    ...
```

Or:

```python
if iterations > 1000:  # Default convergence limit
    ...
```

## What NOT to Comment

- **What the code does** (if the code is clear)
- **Obvious operations** (`i += 1  # increment i`)
- **Repeated information** from docstrings

Comments should explain **why**, not **what**, unless the solution is convoluted or
subtle.

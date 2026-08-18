# Docstring conventions

Use the [contributing guide](../../../../docs/source/development/contributing.md#docstrings)
as the canonical style policy. Use this reference for audit judgment that is specific to
Python docstrings.

## Determine the public contract

Prioritize objects exported through package namespaces, listed in `__all__`, or used in
the user guide and examples. A name without a leading underscore is an audit candidate,
not proof that it is a supported public interface.

Private helpers generally need only enough documentation to explain non-obvious
assumptions. Do not require full public-interface sections for simple accessors or private
implementation details.

## Compare signatures and docstrings

For each public callable:

1. Compare parameter names, order, types, and defaults with the signature.
2. Compare documented returns and exceptions with actual behavior.
3. Confirm array shapes and semantic dimensions against the implementation.
4. Check examples against current imports and construction patterns.

Equivalent type phrasing is acceptable. Examples include `int | None` in a signature and
`int or None` in a NumPy-style docstring, or `list[Callback]` and `list of Callback`.
Report only inconsistencies that could mislead a caller or break rendered documentation.

## Validate content by object type

Use sections only when the object has relevant content:

| Object | Expected content |
|---|---|
| Function or method | Parameters, Returns, and Raises when applicable |
| Class | Constructor parameters and meaningful post-construction attributes |
| Dataclass | Fields in Parameters rather than a duplicate Attributes section |
| Property | A concise summary and any non-obvious type or behavior |

For a single return, document the type without a name. Name each item when a callable
returns multiple values.

## Check references and inline code

- Use Sphinx roles for documented classes, methods, functions, and modules when a
  clickable reference helps the reader.
- Use a fully qualified target where short names are ambiguous. A leading tilde may
  shorten the displayed name.
- Use backticks for parameters, attributes, shapes, and literals.
- Treat searches for snake-case or CamelCase terms only as candidate discovery. Natural
  prose and established scientific notation are not formatting violations.

## Scientific terminology

Define acronyms on first use in a document or self-contained section. Standard symbols
such as `mu`, `sigma`, `alpha`, and `beta` are acceptable when their meaning is clear from
the mathematical context. Use [terminology.md](terminology.md) for project terms and
semantic dimension names.

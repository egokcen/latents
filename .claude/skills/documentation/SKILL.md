---
name: documentation
description: Documentation conventions for the latents project. Use when auditing docstrings, source docs, inline comments, or root-level documentation. Covers NumPy docstring style, Sphinx RST patterns, cross-references, and domain terminology.
---

# Documentation Conventions

This skill defines documentation conventions for the latents project. It serves as a reference for documentation auditors.

## Scope

| Documentation Type | Location | Conventions |
|--------------------|----------|-------------|
| Docstrings | `src/**/*.py` | [Contributing guide](../../../docs/source/development/contributing.md#docstrings) + [docstrings.md](references/docstrings.md) |
| Source docs | `docs/source/**/*.{md,rst}` | [source-docs.md](references/source-docs.md) |
| Inline comments | `src/**/*.py` | [inline-comments.md](references/inline-comments.md) |
| Root docs | `*.md` in repo root | [root-docs.md](references/root-docs.md) |

## Quick Reference

### File Formats

- **MyST Markdown** (`.md`): Prose-heavy content (guides, installation, concepts)
- **reStructuredText** (`.rst`): API reference (autodoc directives)

### Title Style

Use **sentence case** for all titles (capitalize first word and proper nouns only).

### Cross-References

| Link Type | Format |
|-----------|--------|
| Internal doc | `{doc}` or `:doc:` |
| Label ref | `{ref}` or `:ref:` |
| API class | `:class:` |
| API function | `:func:` with tilde |
| External URL | Markdown link |

### Autodoc Options

| Class Type | Pattern |
|------------|---------|
| Standard class | `:members:` |
| Pure data dataclass | `:no-members:` |
| Dataclass with methods | `:members:` + `:exclude-members: field1, field2` |
| Primary user-facing | `:autosummary:` with `:autosummary-sections: Methods` |

**Avoid**: `:show-inheritance:`, `:undoc-members:`, `.. automodule::` on index pages

## Supporting References

- [Check codes](references/check-codes.md) — All audit codes (DOC-*, SRC-*, ROOT-*, CMT-*, VIS-*)
- [Terminology](references/terminology.md) — Domain glossary and variable naming
- [Docstrings](references/docstrings.md) — Audit procedures for Python docstrings
- [Source docs](references/source-docs.md) — RST patterns, index pages, module pages
- [Inline comments](references/inline-comments.md) — Shape comments, TODOs, section headers
- [Root docs](references/root-docs.md) — Code examples, command style, cross-file consistency

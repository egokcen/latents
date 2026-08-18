# Root Documentation Conventions

Conventions for root-level markdown files (`README.md`, `CHANGELOG.md`, etc.).

**Note**: `CONTRIBUTING.md` at the repo root is a redirect stub pointing to the full
contributing guide in `docs/source/development/contributing.md`. Then Sphinx docs are
the source of truth for contributor documentation.

## Code Examples

All code examples in root docs must be runnable.

### Fenced Code Blocks

Use fenced code blocks with language identifier:

````markdown
```bash
uv run pytest
```
````

````markdown
```python
from latents.gfa import GFAModel
model = GFAModel()
```
````

### Shell Commands

Shell commands should use the project's tooling:

| Correct | Incorrect |
|---------|-----------|
| `uv run pytest` | `pytest` |
| `uv run pre-commit run --all-files` | `pre-commit run --all-files` |
| `uv run sphinx-build ...` | `sphinx-build ...` |

**Rationale**: `uv run` ensures the correct virtual environment and dependencies.

### Output Examples

When showing command output, use a separate unfenced block or include it in the same block with clear separation:

````markdown
```bash
uv run pytest tests/test_gfa/test_inference.py -v
```

Output:
```
tests/test_gfa/test_inference.py::test_fit PASSED
tests/test_gfa/test_inference.py::test_infer_latents PASSED
```
````

## Cross-File Consistency

Terms and commands should be consistent across all root docs.

### Commands

If the contributing guide documents a command one way, `README.md` should use the same form:

| Document | Command |
|----------|---------|
| contributing.md | `uv run pre-commit run --all-files` |
| README.md | `uv run pre-commit run --all-files` (same) |

### Project Metadata

These should match across files:

- Package name (`latents`)
- Repository URL
- Author information
- License

### Terminology

Use the same terms as in source documentation. See [terminology.md](terminology.md).

## README Structure

The README should include these sections (in rough order):

| Section | Required | Contents |
|---------|----------|----------|
| Title + badges | Yes | Package name, CI status, version badges |
| Description | Yes | One-paragraph summary |
| Installation | Yes | How to install the package |
| Quick start | Yes | Minimal working example |
| Documentation | Yes | Link to full docs |
| Contributing | Recommended | Link to CONTRIBUTING.md |
| License | Yes | License type and link |

## Contributing Guide Structure

The contributing guide (`docs/source/development/contributing.md`) should include:

| Section | Contents |
|---------|----------|
| Development setup | How to clone, install deps, run tests |
| Code style | Linting, formatting, conventions |
| Testing | How to run tests, add new tests |
| Documentation | How to build docs, docstring style |
| Pull requests | PR process, commit conventions |

## CHANGELOG Format

Follow [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [Unreleased]

### Added
- New feature description

### Changed
- Change to existing functionality

### Fixed
- Bug fix description

## [0.1.0] - 2025-01-15

### Added
- Initial release features
```

### Categories

| Category | Use For |
|----------|---------|
| Added | New features |
| Changed | Changes to existing functionality |
| Deprecated | Soon-to-be removed features |
| Removed | Removed features |
| Fixed | Bug fixes |
| Security | Vulnerability fixes |

## Markdown Style

### Headings

Use ATX-style headings (`#`) not Setext-style (underlines):

```markdown
# Good heading

Bad heading
===========
```

### Lists

Use `-` for unordered lists, `1.` for ordered lists:

```markdown
- Item one
- Item two

1. First step
2. Second step
```

### Links

Prefer inline links for short URLs:

```markdown
See the [documentation](https://latents.readthedocs.io).
```

Use reference links for repeated or long URLs:

```markdown
See the [documentation][docs] and [API reference][api].

[docs]: https://latents.readthedocs.io
[api]: https://latents.readthedocs.io/en/latest/reference/
```

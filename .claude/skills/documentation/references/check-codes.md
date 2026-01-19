# Documentation Audit Check Codes

All check codes used by documentation auditors, organized by category.

## Severity Levels

| Level | Meaning | Action Required |
|-------|---------|-----------------|
| **Error** | Violation that will cause problems (broken links, missing required content) | Must fix before merge |
| **Warning** | Inconsistency or suboptimal practice | Should fix; document exception if not |
| **Info** | Suggestion for improvement | Consider fixing |

---

## Docstring Checks (DOC-*)

Checks for Python docstrings in source code. See `audit-docstring` agent.

### Missing Content (DOC-M*)

| Code | Severity | Issue |
|------|----------|-------|
| `DOC-M001` | Error | Public function or class missing docstring |
| `DOC-M002` | Error | `__init__` missing docstring when class has no class-level docstring |

### Required Sections (DOC-S*)

| Code | Severity | Issue |
|------|----------|-------|
| `DOC-S001` | Error | Missing required section for object type |
| `DOC-S002` | Warning | Returns section uses wrong format (single vs multiple) |

### Type Formatting (DOC-T*)

| Code | Severity | Issue |
|------|----------|-------|
| `DOC-T001` | Warning | Union type uses `\|` instead of `or` in docstring |
| `DOC-T002` | Warning | Missing default value in parameter description |

### Inline Code Formatting (DOC-F*)

| Code | Severity | Issue |
|------|----------|-------|
| `DOC-F001` | Info | Inconsistent backtick style within a file |
| `DOC-F002` | Warning | Code term appears without backticks in prose |
| `DOC-F003` | Warning | Same term formatted inconsistently across codebase |

### Signature Consistency (DOC-C*)

| Code | Severity | Issue |
|------|----------|-------|
| `DOC-C001` | Error | Parameter in docstring not in signature |
| `DOC-C002` | Error | Parameter in signature not in docstring |
| `DOC-C003` | Warning | Parameter order mismatch |
| `DOC-C004` | Warning | Docstring type doesn't match signature type hint |
| `DOC-C005` | Warning | Signature has type hint but docstring omits type |

### Cross-References (DOC-R*)

| Code | Severity | Issue |
|------|----------|-------|
| `DOC-R001` | Warning | Documented class/function in backticks should use Sphinx role |
| `DOC-R002` | Warning | Inconsistent cross-reference style for same object |
| `DOC-R003` | Warning | Module-level function missing full path with tilde |
| `DOC-R004` | Info | Module reference missing full path with tilde |

### Examples (DOC-E*)

| Code | Severity | Issue |
|------|----------|-------|
| `DOC-E001` | Info | Public entry-point missing Examples section |

---

## Source Documentation Checks (SRC-*)

Checks for Sphinx documentation in `docs/source/`. See `audit-source-docs` agent.

### Structure (SRC-S*)

| Code | Severity | Issue |
|------|----------|-------|
| `SRC-S001` | Error | Page not included in any toctree (orphan without `:orphan:` directive) |
| `SRC-S002` | Warning | toctree references non-existent file |
| `SRC-S003` | Warning | Duplicate page in toctree |
| `SRC-S004` | Info | Empty toctree (stub page) |

### Links (SRC-L*)

| Code | Severity | Issue |
|------|----------|-------|
| `SRC-L001` | Error | Broken internal cross-reference |
| `SRC-L002` | Error | Broken external URL |
| `SRC-L003` | Warning | Internal link uses raw URL instead of cross-reference |
| `SRC-L004` | Info | External URL could use intersphinx |

### Titles (SRC-T*)

| Code | Severity | Issue |
|------|----------|-------|
| `SRC-T001` | Warning | Title not in sentence case |
| `SRC-T002` | Warning | Module page title doesn't match module path |
| `SRC-T003` | Info | Inconsistent title style across similar pages |

### RST Patterns (SRC-R*)

| Code | Severity | Issue |
|------|----------|-------|
| `SRC-R001` | Warning | Missing `.. currentmodule::` after `.. automodule::` |
| `SRC-R002` | Warning | Using `:show-inheritance:` (should be avoided) |
| `SRC-R003` | Warning | Using `:undoc-members:` (should be avoided) |
| `SRC-R004` | Info | Large class missing `:autosummary:` for methods |
| `SRC-R005` | Warning | Dataclass with redundant attribute documentation (missing `:no-members:` or `:exclude-members:`) |

### Content (SRC-C*)

| Code | Severity | Issue |
|------|----------|-------|
| `SRC-C001` | Warning | Index page uses `.. automodule::` (causes unintended API entries) |
| `SRC-C002` | Info | Category index missing introductory paragraph |
| `SRC-C003` | Warning | Terminology inconsistency with glossary |

---

## Root Documentation Checks (ROOT-*)

Checks for root-level markdown files. See `audit-root-docs` agent.

### Code Examples (ROOT-E*)

| Code | Severity | Issue |
|------|----------|-------|
| `ROOT-E001` | Error | Code example fails to run |
| `ROOT-E002` | Warning | Shell command missing `uv run` prefix |
| `ROOT-E003` | Warning | Code block missing language identifier |

### Consistency (ROOT-C*)

| Code | Severity | Issue |
|------|----------|-------|
| `ROOT-C001` | Warning | Command differs between files (e.g., README vs CONTRIBUTING) |
| `ROOT-C002` | Warning | Version or URL mismatch between files |
| `ROOT-C003` | Info | Terminology inconsistency with source docs |

### Structure (ROOT-S*)

| Code | Severity | Issue |
|------|----------|-------|
| `ROOT-S001` | Warning | README missing expected section |
| `ROOT-S002` | Info | CHANGELOG entry missing for recent change |

---

## Inline Comment Checks (CMT-*)

Checks for inline comments in source code. See `audit-comments` agent.

### Shape Comments (CMT-S*)

| Code | Severity | Issue |
|------|----------|-------|
| `CMT-S001` | Warning | Complex broadcasting operation missing shape comment |
| `CMT-S002` | Info | Shape comment format inconsistent |

### TODO/FIXME (CMT-T*)

| Code | Severity | Issue |
|------|----------|-------|
| `CMT-T001` | Info | TODO without description |
| `CMT-T002` | Info | FIXME without description |
| `CMT-T003` | Warning | Stale TODO (references closed issue) |

### Magic Numbers (CMT-M*)

| Code | Severity | Issue |
|------|----------|-------|
| `CMT-M001` | Warning | Magic number without explanatory comment |

---

## Visual Documentation Checks (VIS-*)

Checks for rendered HTML documentation. See `audit-visual` agent.

### Rendering (VIS-R*)

| Code | Severity | Issue |
|------|----------|-------|
| `VIS-R001` | Error | Page fails to render |
| `VIS-R002` | Error | Math rendering failure |
| `VIS-R003` | Warning | Broken image |
| `VIS-R004` | Warning | Missing gallery thumbnail |

### Layout (VIS-L*)

| Code | Severity | Issue |
|------|----------|-------|
| `VIS-L001` | Warning | Signature overflows container |
| `VIS-L002` | Info | Table too wide for viewport |

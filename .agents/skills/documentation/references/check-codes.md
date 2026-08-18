# Documentation audit check codes

Use these stable rule codes in documentation audits. Assign a separate sequential finding
identifier such as `DOC-001` to each reported occurrence or consolidated root cause.

The listed severity is the default. Adjust it only when concrete impact justifies the
change, and explain the adjustment.

## Docstrings

| Rule | Severity | Report when |
|---|---|---|
| `DOC-CONT-001` | High | An exported public object lacks documentation needed to use it safely |
| `DOC-CONT-002` | Medium | A public contract omits a material parameter, return value, attribute, exception, or example |
| `DOC-SIG-001` | High | A documented parameter or return value contradicts the callable signature or behavior |
| `DOC-SIG-002` | Medium | Parameter order, type, or default is materially inconsistent with the signature |
| `DOC-XREF-001` | Medium | A code-object reference is broken, ambiguous, or resolves to the wrong object |
| `DOC-TERM-001` | Low | Terminology is materially inconsistent with the project glossary |

## Sphinx source documentation

| Rule | Severity | Report when |
|---|---|---|
| `SRC-BUILD-001` | High | The warning-as-error Sphinx build fails because of project documentation |
| `SRC-TREE-001` | High | A page is unintentionally orphaned or a toctree target does not exist |
| `SRC-TREE-002` | Medium | A page appears more than once or in a misleading navigation location |
| `SRC-LINK-001` | High | An internal cross-reference is broken or resolves incorrectly |
| `SRC-LINK-002` | Medium | An external link is confirmed broken rather than transiently unavailable |
| `SRC-RST-001` | Medium | A reStructuredText directive causes missing, duplicate, or misleading rendered content |
| `SRC-CONT-001` | Medium | A page conflicts with the public interface or another canonical project document |
| `SRC-TITLE-001` | Low | A title materially violates the sentence-case or module-name convention |

## Root documentation

| Rule | Severity | Report when |
|---|---|---|
| `ROOT-EXEC-001` | High | A documented example or command is confirmed not to run as written |
| `ROOT-CONS-001` | Medium | Commands, versions, URLs, or behavior conflict across canonical documents |
| `ROOT-CONT-001` | Medium | A user-facing root document omits information necessary for its stated purpose |
| `ROOT-CHG-001` | Medium | A user-facing change lacks the required changelog entry |
| `ROOT-FMT-001` | Low | Markdown formatting impairs rendering, copying, or comprehension |

## Inline comments

| Rule | Severity | Report when |
|---|---|---|
| `CMT-SHAPE-001` | Medium | A non-obvious array transformation lacks the shape or semantic context needed to review it |
| `CMT-TASK-001` | Low | A TODO or FIXME is vague, stale, or references a confirmed closed issue |
| `CMT-NUM-001` | Low | A scientifically meaningful threshold or numerical constant lacks rationale |
| `CMT-STALE-001` | Medium | A comment contradicts the code and could mislead maintenance |

## Rendered documentation

| Rule | Severity | Report when |
|---|---|---|
| `VIS-PAGE-001` | High | A user-facing source page has no corresponding rendered page |
| `VIS-MATH-001` | High | Mathematical notation is visibly unrendered or incorrect |
| `VIS-ASSET-001` | High | A required image or gallery asset is broken or missing |
| `VIS-LAYOUT-001` | Low | Direct visual evidence shows overflow, clipping, or unusable responsive layout |

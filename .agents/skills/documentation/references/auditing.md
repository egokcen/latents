# Documentation audit procedure

Apply this procedure to every documentation audit, regardless of scope.

## 1. Establish scope

- Record the requested target, files inspected, relevant source files, exclusions, and
  existing rendered artifacts.
- Treat generated documentation such as `docs/source/auto_examples/` according to its
  generator rather than auditing generated text as hand-written source.
- Read the relevant project configuration before assuming a generic Sphinx, MyST, or
  reStructuredText convention applies.

## 2. Gather evidence

- Use searches only to identify candidates. Inspect the surrounding documentation,
  source code, configuration, or rendered output before reporting a finding.
- Prefer deterministic evidence: warning-as-error builds, link checks, toctrees, callable
  signatures, and direct rendered inspection.
- Do not claim that a command passed unless its exit status was observed. If permissions
  prevent a build or the build is stale, record that limitation.
- Confirm external-link failures before reporting them. A single network error is not
  sufficient evidence.

## 3. Judge materiality

- Report user impact, correctness, build reliability, or material maintainability issues.
- Do not report preferences that are absent from project policy.
- Consolidate repeated symptoms with one root cause. List representative locations and
  state the broader scope.
- Record intentional exceptions so later audit passes do not repeatedly report them.

## 4. Assign severity and confidence

Use one severity scale across audit scopes:

- **Blocker**: a demonstrated merge or release blocker
- **High**: likely incorrect output, broken documentation, or user-facing failure
- **Medium**: material ambiguity, inconsistency, or likely future failure
- **Low**: localized improvement with limited impact

Do not use informational observations as findings. Put useful limitations or context in
the scope or residual-risk section.

Assign confidence independently:

- **High**: directly reproduced or proven from authoritative source and configuration
- **Medium**: strong evidence remains but one relevant condition could not be verified
- **Low**: do not report by default; investigate further or record as a limitation

## 5. Return the audit

Return findings to the requesting agent or user. Do not write an audit file unless the
user explicitly requests persistence.

Use stable rule codes from [check-codes.md](check-codes.md), plus sequential finding
identifiers local to the response:

```markdown
## Scope

- Target:
- Files inspected:
- Source files cross-referenced:
- Commands run:
- Exclusions or limitations:

## Findings

### DOC-001: Brief title

- Rule: DOC-SIG-001
- Severity: High
- Confidence: High
- Location: `path/to/file.py:42`
- Evidence: Concise, directly observed evidence.
- Impact: Concrete consequence for users or maintainers.
- Recommendation: Specific next action.

## Residual risk

- Areas not evaluated:
```

Omit the residual-risk section when the scope is complete. If there are no material
findings, say so directly and retain the scope and limitations. Add a severity summary
table only when it materially improves navigation through several findings.

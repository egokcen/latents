# Test audit procedure

## 1. Establish scope

- Record the requested tests, source files cross-referenced, commands run, exclusions,
  and relevant coverage data.
- Inspect `conftest.py` files and helper modules that affect the selected tests.
- Map tests to behavior through imports, fixtures, and call paths. Do not assume that
  matching filenames prove coverage.

## 2. Gather evidence

- Read both the test and the implementation before judging adequacy.
- Run the smallest relevant test or coverage command when permissions allow. Record the
  command and exit status. Do not imply that a command ran when it did not.
- Use searches only to identify candidates. A regex match is not a finding.
- Check whether indirect integration coverage already protects a public contract.
- Consolidate repeated symptoms under one root-cause finding.

## 3. Check registry

The listed severity is the default. Adjust it only when demonstrated impact warrants a
change, and explain the adjustment.

### Numerical behavior

| Rule | Severity | Report when |
|---|---|---|
| `TST-NUM-001` | High | A tolerance can conceal a meaningful regression or reject valid numerical variation |
| `TST-NUM-002` | Medium | Computed floating-point values are compared exactly without a contract requiring exactness |
| `TST-NUM-003` | Medium | A test omits a material invariant such as finiteness, symmetry, positive definiteness, shape, or data type |
| `TST-NUM-004` | Medium | A convergence property is asserted or assumed without support from the algorithm contract |

Prefer `numpy.testing.assert_allclose` for ordinary floating-point comparisons. NumPy
recommends it over decimal-place assertions for consistent comparisons; do not describe
`assert_almost_equal` as deprecated. Require an absolute tolerance near zero when the
intended contract allows nonzero numerical error. Judge tolerances from the operation,
scale, data type, and statistical variability rather than fixed global thresholds.

### Randomness and reproducibility

| Rule | Severity | Report when |
|---|---|---|
| `TST-RNG-001` | Medium | A test depends on NumPy's legacy global random-number state |
| `TST-RNG-002` | High | Random behavior can produce an irreproducible failure or bypass the intended assertion |
| `TST-RNG-003` | Medium | A seed or generator is not propagated through the behavior being tested |

A fixed seed is one reproducibility mechanism, not a universal requirement. Frameworks
that record failing randomized inputs are acceptable. Prefer explicit
`numpy.random.Generator` ownership and reproducible failure evidence.

### Oracles and behavioral coverage

| Rule | Severity | Report when |
|---|---|---|
| `TST-ORC-001` | High | The expected result calls the implementation under test or duplicates its algorithm closely enough to share the defect |
| `TST-COV-001` | High | A test's name or stated contract is not actually verified |
| `TST-COV-002` | Medium | A value-producing behavior is protected only by a shape or no-exception assertion |
| `TST-COV-003` | Medium | Source and coverage evidence show a material public behavior or branch is unprotected |

Do not require one direct test for every public method. Indirect coverage is sufficient
when failures would be localized and the contract is asserted clearly. Plotting smoke
tests are acceptable when they inspect the relevant Matplotlib objects or state.

### Isolation, failures, and runtime

| Rule | Severity | Report when |
|---|---|---|
| `TST-ISO-001` | High | Shared mutable fixtures, global state, or test ordering can change outcomes |
| `TST-ISO-002` | Medium | A test relies unnecessarily on the working directory, wall-clock timing, or files outside pytest temporary paths |
| `TST-ERR-001` | Medium | A material documented error or warning path is untested |
| `TST-PERF-001` | Medium | A test or fixture fits a model to convergence without the `fit` marker |
| `TST-STR-001` | Low | Repetition or control flow materially obscures which cases ran or allows zero assertions |

Do not require `match` on every `pytest.raises` call. Require it when message text is part
of the contract or distinguishes multiple branches with the same exception type. Loops,
conditionals, classes, and section comments are not findings by themselves. Leave unused
imports and formatting to Ruff.

## 4. Assign confidence

- **High**: directly reproduced or proven from the test and production path
- **Medium**: strong evidence remains but one relevant condition could not be verified
- **Low**: investigate further or record as a limitation rather than a finding

Use one severity scale:

- **Blocker**: a demonstrated merge or release blocker
- **High**: a likely escaped correctness failure or seriously unreliable test signal
- **Medium**: a material coverage, reproducibility, or maintainability weakness
- **Low**: a localized improvement with limited risk

Do not create informational findings. Put useful context in the scope or residual-risk
section.

## 5. Return findings

Use sequential finding identifiers local to the response and stable rule codes from the
registry above. Return the response instead of writing an audit file.

```markdown
## Scope

- Target:
- Test files inspected:
- Source files cross-referenced:
- Commands run:
- Exclusions or limitations:

## Findings

### TST-001: Brief title

- Rule: TST-ORC-001
- Severity: High
- Confidence: High
- Location: `tests/path/test_module.py:42`
- Evidence: Concise evidence from the test and production path.
- Impact: Concrete regression that could escape or valid behavior that could fail.
- Recommendation: Specific next action.

## Residual risk

- Areas not evaluated:
```

If there are no material findings, say so directly and retain the scope and limitations.
Add a severity summary only when it improves navigation through several findings.

---
name: audit-tests
description: Audit Latents tests and fixtures for scientific correctness, numerical assertions, reproducibility, oracle independence, behavioral coverage, isolation, failure paths, and fit-test classification. Use for read-only test-quality reviews that cross-reference tests with production code. Do not use to write tests, fix failures, or review documentation.
---

# Audit tests

Perform a read-only, evidence-based review. Return findings to the requesting agent or
user. Do not edit tests, source code, or audit files unless the user explicitly expands
the task.

Before auditing:

1. Read the testing policy in
   [`docs/source/development/contributing.md`](../../../docs/source/development/contributing.md#testing).
2. Read [audit-procedure.md](references/audit-procedure.md) for the check registry,
   scientific-testing heuristics, and response format.

Cross-reference every test candidate with the production behavior it claims to protect.
Treat searches, coverage percentages, and filename mappings as leads rather than proof.

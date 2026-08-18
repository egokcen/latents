# Rendered documentation conventions

Audit rendered HTML only when a current build is available or the requesting agent can
produce one with:

```sh
uv run sphinx-build -W -b html docs/source docs/_build/html
```

If the build cannot be run, state the build age or absence as a limitation. Do not infer
visual correctness from source markup alone.

## Inspection priorities

Inspect representative user journeys and page types:

- project landing page
- installation or getting-started page
- principal example gallery page
- main model API page
- a dataclass or configuration API page

For every reported visual finding, capture direct evidence from the rendered page or
HTML. Check:

- every intended source page has a rendered destination
- mathematical notation is rendered rather than exposed as raw LaTeX
- local image and gallery references resolve to existing assets
- signatures, tables, code blocks, and navigation remain usable at representative widths

Long signatures, large tables, and raw LaTeX-like strings are candidates, not findings.
Report layout problems only after observing overflow, clipping, or unusable navigation.

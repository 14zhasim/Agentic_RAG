# FinanceBench package

This directory contains the golden-path implementation:

```text
config.py      load and validate TOML settings
data.py        prepare, validate and load FinanceBench
conditions.py  construct the five context conditions
generation.py  build prompts and call OpenRouter
metrics.py     calculate one job's retrieval metrics
reporting.py   aggregate saved predictions
cli.py         route terminal commands to those modules
execution/
├── preflight.py  inspect jobs without API calls
├── job.py        execute one question-condition pair
└── runner.py     loop, checkpoint and resume real runs
```

For the authoritative explanation, diagrams, data contracts, public-function
call order, tests, and as-built review slices, read:

`docs sys design/benchmark/FinanceBench Implementation Guide.md`

For requirements and approved design decisions, read:

`docs sys design/Benchmark.md`

This README deliberately does not duplicate their requirements or implementation
decisions.

# FinanceBench package

This directory contains the golden-path implementation:

```text
config.py                         load and validate TOML settings
cli.py                            route terminal commands to the modules
dataset/
├── financebench.py              prepare, validate and load FinanceBench
└── subsets.py                   select fixed development subsets
pipeline/
├── conditions.py                construct the five context conditions
└── generation.py                build prompts and call OpenRouter
execution/
├── preflight.py  inspect jobs without API calls
├── job.py        execute one question-condition pair
└── runner.py     loop, checkpoint and resume real runs
evaluation/
├── retrieval_metrics.py         calculate one job's retrieval metrics
├── judge.py                     judge saved answers
├── judge_validation.py          validate the judge against published labels
└── manual_review.py             adjudicate disputed judgments
reporting/
├── report.py                    aggregate saved predictions
├── failure_analysis.py          diagnose retrieval failures
└── workbook.py                  render the report workbook
```

For the authoritative explanation, diagrams, data contracts, public-function
call order, tests, and as-built review slices, read:

`docs sys design/benchmark/FinanceBench Implementation Guide.md`

For requirements and approved design decisions, read:

`docs sys design/Benchmark.md`

This README deliberately does not duplicate their requirements or implementation
decisions.

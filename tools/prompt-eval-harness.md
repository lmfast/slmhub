---
title: Prompt & Eval Harness
description: A tiny evaluation harness to keep you honest about regressions.
---

# Prompt & eval harness (lightweight)

## Why this exists

Benchmarks don’t represent your workload. You need a tiny suite that does.

## What to build

- `prompts/acceptance.jsonl`: real prompts
- `prompts/adversarial.jsonl`: injection and weird edge cases
- `grading.md`: rubric (or unit tests if outputs are structured)

## Success criteria

- reproducible across revisions,
- catches regressions from quantization/runtime changes,
- cheap enough to run on every PR.



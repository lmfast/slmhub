---
title: Model page standard
description: "A consistent, developer-centric template for model pages."
---


# Model page standard

## Required sections (in this order)

1. **Quickstart**: 1–2 runnable examples (Transformers + one local runtime).
2. **When to use**: task fit and deployment fit.
3. **What to watch out for**: failure modes, license notes, safety concerns.
4. **Metadata**: model id, license, last modified.

## What to avoid

- Benchmark tables meant to “win”.
- Claims without reproduction steps.
- Long marketing intros.

## If you add a hand-written model page

Prefer placing it in `models/featured/` and keep it stable. If it’s data-driven, generate it under `models/generated/`.



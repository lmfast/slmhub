---
title: Edge-first, cloud fallback
description: "Run locally/edge first; fallback to cloud only when needed."
---


# Edge-first, cloud fallback

## Why it works

- Most requests are “easy enough” for an SLM.
- Sensitive data stays local by default.
- Costs are predictable.

## Implementation sketch

- Classify requests by complexity and risk.
- Attempt edge/local inference.
- If confidence is low, or latency is too high, fallback to a larger model (or human review).

## Operational advice

- Log only what you need (privacy).
- Add a kill switch for fallback.
- Track fallback rate as a primary KPI.



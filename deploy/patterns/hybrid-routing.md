---
description: Use an SLM as default and route to a larger model only when needed.
---

# Hybrid routing (SLM + LLM)

## The objective

Maximize:

- quality for hard tasks,
- cost and latency efficiency for the majority.

## A simple router

Route based on:

- request type (summarization vs codegen vs extraction),
- context length required,
- compliance constraints,
- confidence signals (self-checks, heuristics).

## Minimum viable implementation

- Start with manual rules.
- Add a small classifier only if rules become too complex.
- Track: **fallback rate**, **P95 latency**, **cost per request**, **error rate**.



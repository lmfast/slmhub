---
description: The editorial and technical principles that keep this docs useful to developers.
---

# Principles

## Developer-first, not hype-first

- **Show working code** first.
- **State constraints** early (VRAM/RAM, latency sensitivity, privacy).
- **Explain tradeoffs** in plain language.

## Discovery over leaderboards

We intentionally avoid “model X beats model Y” pages and competitive benchmark tables.

Instead, we focus on:

- **What a model is good at** (task fit, languages, reasoning style).
- **How to run it** (frameworks, quantization, serving patterns).
- **How to validate it** for your own workload (small eval harness + acceptance tests).

## Minimal UI, maximum readability

GitBook gives clean typography by default. Our job is to keep pages:

- short sections,
- concrete headings,
- scannable “Quickstart” blocks,
- honest “Known pitfalls”.

## Always current (automated where possible)

- “Directory” pages are **generated** from upstream metadata.
- Links and examples are **checked** in CI.
- For deeper claims, we prefer **references** and “how to reproduce”.



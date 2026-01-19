---
title: Cost Controls
description: How to keep unit economics predictable.
---

# Cost controls

## The big cost drivers

- tokens (input + output),
- retries and timeouts,
- fallback rate to larger models,
- context window bloat (RAG that retrieves too much).

## Controls that work

- strict max tokens per request,
- caching (prompt-prefix or retrieval cache),
- hybrid routing with hard budgets,
- nightly regressions on acceptance prompts.



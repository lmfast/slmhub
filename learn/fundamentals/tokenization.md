---
description: Tokenization, only what you need to debug costs and context.
---

# Tokenization (just enough)

## Why developers should care

- **Cost & latency scale with tokens**, not characters.
- Tokenization determines **context usage** and can silently break prompts (especially multilingual).

## Practical checks

- **Count tokens** for your typical prompts.
- Test representative text in your domain (code, logs, JSON, languages).
- If a prompt “should fit” but doesn’t, it’s often tokenization + overhead (system prompt, tools schema, retrieved context).

## Rule of thumb

If you’re doing RAG, reserve:

- \(20\%-40\%\) of context for retrieval,
- \(5\%-15\%\) for system/tooling overhead,
- the rest for user prompt + model output.



---
description: A developer’s mental model for when small models win.
---

# SLMs vs LLMs

## The decision in one minute

Choose an SLM first if:

- you need **low latency**,
- you need **data locality** (on-device/on-prem),
- you need **cost control**,
- you can accept **narrower capability** and will validate on your own tasks.

Choose a large model first if:

- you need broad, “generalist” capability,
- your task is high-stakes and you can’t tolerate many retries,
- you can’t control input quality and need robustness.

## Common winning patterns for SLMs

- **RAG + SLM**: retrieval handles “knowledge”, model handles “reasoning over context”.
- **Hybrid routing**: SLM for 80–95% of requests, LLM fallback for the rest.
- **Specialists**: several small task-specific models outperform a single general model operationally.

## How to validate quickly

Build a tiny “acceptance test” set:

- 20–50 prompts that represent your real workload,
- expected outputs or grading rubrics,
- run it on every model change (new revision, new quantization).

See `tools/prompt-eval-harness.md`.



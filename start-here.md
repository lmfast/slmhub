---
description: The fastest path to using SLM Hub as a developer.
---

# Start here

## If you want results in 10 minutes

- **Run a model locally**: `deploy/quickstarts/ollama.md`
- **Serve an API**: `learn/tutorials/serve-openai-compatible.md`
- **Pick a model for your use case**: `tools/model-picker.md`

## What “Small Language Model” means in this docs

We use **SLM** to mean models that are small enough to be:

- **cheap to run** (often on a single GPU or even CPU),
- **fast to respond** (interactive latency),
- **deployable close to data** (edge/on-prem),
- **easy to fine-tune** with PEFT techniques.

This is intentionally a **use-case definition**, not a strict parameter-count cutoff.

## How the docs stay current

Some pages are hand-written (tutorials, patterns). Some are **auto-generated** from Hugging Face metadata, especially the model directory.

- **Policy**: `community/update-policy.md`
- **Automation**: `scripts/` + `.github/workflows/`



---
title: Model picker
description: "A simple decision framework to pick a model without chasing leaderboards."
---


## Step 1: constraints

- **Where will it run?** (browser / edge / single GPU / multi GPU)
- **Latency target?**
- **Privacy/compliance?**
- **Languages?**
- **Context length?**

## Step 2: choose a baseline runtime

- easiest local dev: **Ollama**
- production throughput: **vLLM**
- edge/CPU: **llama.cpp**
- browser demos: **Transformers.js**

## Step 3: shortlist 3 models

- 1 “safe default” (widely used)
- 1 “smaller/faster” candidate
- 1 “quality-first” candidate

Use `models/generated/directory.md` for discovery.

## Step 4: acceptance prompts

Create a small eval set:

- 20–50 real prompts
- expected outputs or a grading rubric

Then run it on:

- different model revisions,
- different quantization settings,
- different runtimes.

See `tools/prompt-eval-harness.md`.



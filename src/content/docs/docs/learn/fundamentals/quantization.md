---
title: Quantization (practical)
description: "How to make models fit and run fast without getting surprised."
---


## What you’re trading

Quantization reduces memory and can improve throughput, but may change behavior:

- **Lower bits** → cheaper + faster, but sometimes less stable outputs.
- **Different runtimes** (llama.cpp vs vLLM vs Transformers) behave differently under the same bit-width.

## Use this decision path

- **Need fastest “ship it” local dev**: start with **Ollama**.
- **Need highest throughput** on GPUs: use **vLLM** (and then choose a quant format supported by your GPU stack).
- **Need edge/CPU**: use **llama.cpp** with GGUF.

See `deploy/quickstarts/`.

## Don’t break production by accident

- Pin **model revision** (commit hash) in production.
- Pin **quantization format** and **runtime version**.
- Run your acceptance tests when you change any of the above.



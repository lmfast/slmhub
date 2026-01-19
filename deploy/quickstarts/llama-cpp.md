---
title: llama.cpp
description: Portable inference for CPU/edge using llama.cpp.
---

# llama.cpp

## Why use it

- Runs on many environments (CPU, edge, constrained devices).
- Works well with GGUF quantized weights.

## Practical workflow

- Get a GGUF build of your model (or convert).
- Run the CLI or server mode.

## Notes

llama.cpp is great for edge and single-user flows. If you need high-throughput multi-user serving, use `deploy/quickstarts/vllm.md`.



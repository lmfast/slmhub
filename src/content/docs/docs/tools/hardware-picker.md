---
title: Hardware picker
description: "Pick hardware based on workload shape rather than model hype."
---


## Inputs you need

- average tokens in/out
- target P95 latency
- peak concurrency
- memory constraints (VRAM/RAM)

## Practical defaults

- **Development**: whatever you have + quantization.
- **Single-user edge**: llama.cpp + GGUF + aggressive quantization.
- **Multi-user server**: vLLM + GPU(s) + continuous batching.

## Rule of thumb

If you don’t have traffic data, start with a conservative budget and measure:

- tokens/sec,
- latency distribution,
- OOM rate.



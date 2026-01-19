---
title: Scaling & Batching
description: "Throughput and latency controls: batching, caching, timeouts, and capacity planning."
---

# Scaling & batching

## Throughput levers

- **Continuous batching** (runtime-dependent)
- **KV cache** sizing
- **Max tokens** limits
- **Timeouts** and circuit breakers

## Capacity planning (practical)

Start with real traffic traces:

- average and worst-case prompt length,
- target P95 latency,
- concurrency distribution.

Then choose:

- one runtime (vLLM for throughput),
- a stable quantization + model revision,
- conservative limits, and widen slowly.



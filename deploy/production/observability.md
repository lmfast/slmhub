---
description: What to measure so you can debug cost, latency, and quality.
---

# Observability

## Metrics to track

- Latency (P50/P95/P99)
- Tokens in/out
- Requests/sec
- GPU/CPU utilization
- Error rate (timeouts, OOM, upstream failures)
- Fallback rate (if hybrid routing)

## Logs (privacy-aware)

Avoid logging raw user inputs by default. Prefer:

- hashes,
- sampled traces with consent,
- redaction pipelines.



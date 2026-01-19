---
title: vLLM
description: High-throughput GPU serving with an OpenAI-compatible API.
---

# vLLM

## Why use it

- Designed for concurrency and throughput.
- Common choice for production GPU serving.

## Serve (OpenAI-compatible)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000
```

## Smoke test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "List 3 SLM deployment patterns."}],
    "temperature": 0.2
  }'
```

## Production checklist

- Pin model revision.
- Add batching + timeouts.
- Add request logging (privacy-aware).



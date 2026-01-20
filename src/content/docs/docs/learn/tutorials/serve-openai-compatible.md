---
title: Deploy an OpenAI-compatible endpoint
description: "Serve an OpenAI-compatible API locally so your apps can swap providers."
---


## Why this matters

Most tooling expects an OpenAI-style API surface. If you serve locally with that interface, you can:

- swap models without rewriting clients,
- route between SLM/LLM backends,
- centralize observability and safety controls.

## Option A: Ollama (fastest)

Ollama exposes an OpenAI-compatible endpoint in many setups.

- Follow `deploy/quickstarts/ollama.md` for how to run and connect.

## Option B: vLLM (production throughput)

Use vLLM when you care about concurrency and throughput.

- Follow `deploy/quickstarts/vllm.md`.

## Minimal client smoke test

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi4",
    "messages": [{"role": "user", "content": "Say hello in 5 words."}],
    "temperature": 0.2
  }'
```

## Next steps

- Add routing: `deploy/patterns/hybrid-routing.md`
- Add RAG: `learn/tutorials/rag-with-slm.md`



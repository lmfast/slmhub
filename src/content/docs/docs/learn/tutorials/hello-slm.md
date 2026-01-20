---
title: Hello SLM (no setup)
description: "A \"no setup\" first run using browser runtimes and minimal tooling."
---


## Goal

Get a first successful generation with minimal friction. If you want local inference, jump to `learn/tutorials/run-locally-ollama.md`.

## Option A: In-browser inference (WebGPU)

If your browser supports WebGPU, you can run smaller models client-side. A popular path is `@huggingface/transformers` for web.

> This page intentionally stays high-level because browser support and models change quickly; see the official Transformers.js docs referenced from `deploy/quickstarts/transformers-js.md`.

## Option B: Hosted inference (Hugging Face)

If you just want to test a model fast, use HF Inference with a token.

- Create a token in your Hugging Face settings.
- Store it as `HF_TOKEN`.

Then use the code shown in `deploy/quickstarts/transformers-js.md` (web) or `learn/tutorials/run-transformers-python.md` (Python).



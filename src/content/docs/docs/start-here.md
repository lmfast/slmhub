---
title: Start here
description: "The fastest path to using SLM Hub as a developer."
---

## What are Small Language Models?

**Small Language Models (SLMs)** are a class of language models designed to be efficient, accessible, and practical for real-world deployment. Unlike their larger counterparts (LLMs), SLMs prioritize resource efficiency while maintaining strong performance on targeted tasks.

## Defining Characteristics of SLMs

### 1. **Resource Efficiency**

SLMs are designed to run on constrained hardware:

- **Consumer-grade GPUs** (4-16GB VRAM)
- **CPUs** with reasonable RAM (8-32GB)
- **Edge devices** (smartphones, IoT devices, embedded systems)
- **Browser environments** via WebGPU/WebAssembly

### 2. **Fast Inference**

Small models deliver:

- **Low latency** responses (milliseconds to seconds)
- **Interactive experiences** without noticeable delays
- **Real-time applications** (chatbots, autocomplete, live translation)

### 3. **Cost-Effective Deployment**

- **No cloud dependency** required
- **Lower operational costs** (electricity, infrastructure)
- **Affordable fine-tuning** with PEFT techniques (LoRA, QLoRA)

### 4. **Privacy & Data Sovereignty**

- **On-premise deployment** keeps data local
- **Edge processing** eliminates data transmission
- **Compliance-friendly** for regulated industries (healthcare, finance)

### 5. **Task-Specific Excellence**

Rather than being general-purpose, SLMs often excel at:

- **Domain-specific tasks** (code generation, summarization, Q&A)
- **Specialized languages** (non-English, programming languages)
- **Focused reasoning** (math, logic, instruction-following)

## Size is Relative: A Use-Case Definition

We intentionally avoid strict parameter-count cutoffs. An SLM is defined by **what it enables**:

- Can it run where you need it? (laptop, server, edge device)
- Does it respond fast enough? (interactive vs. batch)
- Can you afford to fine-tune it? (PEFT-friendly)
- Does it fit your privacy/compliance needs? (on-prem capable)

**Typical range**: 100M to 10B parameters, but context matters more than numbers.

## Why SLMs Matter in 2026

1. **Democratized AI**: Anyone with a laptop can run powerful language models
2. **Sustainable AI**: Lower energy consumption and carbon footprint
3. **Practical AI**: Faster iteration cycles for developers
4. **Sovereign AI**: Organizations control their data and models
5. **Specialized AI**: Better performance on narrow tasks than general-purpose giants

## Next Steps

- **Explore Models**: Browse our [Model Directory](/slmhub/docs/models/generated/directory/)
- **Deploy Locally**: Try [Ollama Quickstart](/slmhub/docs/deploy/quickstarts/ollama/)
- **Learn Fundamentals**: Start with [SLM vs LLM](/slmhub/docs/learn/fundamentals/slm-vs-llm/)

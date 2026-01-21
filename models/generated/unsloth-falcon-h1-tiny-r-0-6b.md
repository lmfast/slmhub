---
title: Falcon-H1-Tiny-R-0.6B
description: Discovery-focused notes and runnable examples for `unsloth/Falcon-H1-Tiny-R-0.6B`.
---

# Falcon-H1-Tiny-R-0.6B

Discovery-focused notes and runnable examples for `unsloth/Falcon-H1-Tiny-R-0.6B`.

## Quickstart

### Python (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "unsloth/Falcon-H1-Tiny-R-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

prompt = "Explain small language models in 3 bullets."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Local dev (Ollama)

If there’s an Ollama-friendly name for this model in your environment, you can run:

```bash
ollama run falcon-h1-tiny-r-0.6b
```

> If that doesn’t work, use the Transformers path above or see `deploy/quickstarts/ollama.md` for the general workflow.

## When to use this model


- General text generation and assistant-style prompts.

- You need lower latency and lower cost than a large model.


## What to watch out for


- License is unknown in metadata; verify the model card before production use.

- Treat upstream metadata as hints; validate behavior on your own prompts and data.

- If you see quality regressions, pin a model revision instead of tracking `main`.

- Quantization can change behavior; re-run acceptance tests after changing dtype/bit-width.


## Metadata (from Hugging Face)

| Field | Value |
|---|---|
| **Model ID** | `unsloth/Falcon-H1-Tiny-R-0.6B` |
| **Author** | unsloth |
| **Pipeline tag** | text-generation |
| **Library** | transformers |
| **License** | unknown |
| **Last modified** | 2026-01-21T12:14:33+00:00 |
| **Downloads** | 10 |
| **Likes** | 0 |

## Links

- [🤗 Hugging Face model page](https://huggingface.co/unsloth/Falcon-H1-Tiny-R-0.6B)
- [Model card](https://huggingface.co/unsloth/Falcon-H1-Tiny-R-0.6B#model-card)

---

*This page is auto-generated from Hugging Face Hub metadata. If something looks wrong, please open a PR or issue.*
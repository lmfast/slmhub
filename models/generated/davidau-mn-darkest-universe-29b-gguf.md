---
title: MN-DARKEST-UNIVERSE-29B-GGUF
description: Discovery-focused notes and runnable examples for `DavidAU/MN-DARKEST-UNIVERSE-29B-GGUF`.
---

# MN-DARKEST-UNIVERSE-29B-GGUF

Discovery-focused notes and runnable examples for `DavidAU/MN-DARKEST-UNIVERSE-29B-GGUF`.

## Quickstart

### Python (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "DavidAU/MN-DARKEST-UNIVERSE-29B-GGUF"
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
ollama run mn-darkest-universe-29b-gguf
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
| **Model ID** | `DavidAU/MN-DARKEST-UNIVERSE-29B-GGUF` |
| **Author** | DavidAU |
| **Pipeline tag** | text-generation |
| **Library** | — |
| **License** | unknown |
| **Last modified** | 2026-01-21T12:05:54+00:00 |
| **Downloads** | 4.4K |
| **Likes** | 60 |

## Links

- [🤗 Hugging Face model page](https://huggingface.co/DavidAU/MN-DARKEST-UNIVERSE-29B-GGUF)
- [Model card](https://huggingface.co/DavidAU/MN-DARKEST-UNIVERSE-29B-GGUF#model-card)

---

*This page is auto-generated from Hugging Face Hub metadata. If something looks wrong, please open a PR or issue.*
---
title: "my-first-medical-doctor"
description: Discovery-focused notes and runnable examples for `LluisTaha/my-first-medical-doctor`.
---

# my-first-medical-doctor

Discovery-focused notes and runnable examples for `LluisTaha/my-first-medical-doctor`.

## Quickstart

### Python (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "LluisTaha/my-first-medical-doctor"
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
ollama run my-first-medical-doctor
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
| **Model ID** | `LluisTaha/my-first-medical-doctor` |
| **Author** | LluisTaha |
| **Pipeline tag** | text-generation |
| **Library** | transformers |
| **License** | unknown |
| **Last modified** | 2026-01-22T11:38:01+00:00 |
| **Downloads** | 20 |
| **Likes** | 0 |

## Links

- [🤗 Hugging Face model page](https://huggingface.co/LluisTaha/my-first-medical-doctor)
- [Model card](https://huggingface.co/LluisTaha/my-first-medical-doctor#model-card)

---

*This page is auto-generated from Hugging Face Hub metadata. If something looks wrong, please open a PR or issue.*
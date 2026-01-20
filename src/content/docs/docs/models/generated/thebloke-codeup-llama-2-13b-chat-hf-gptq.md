---
title: CodeUp-Llama-2-13B-Chat-HF-GPTQ
description: "Discovery-focused notes and runnable examples for `TheBloke/CodeUp-Llama-2-13B-Chat-HF-GPTQ`."
---


Discovery-focused notes and runnable examples for `TheBloke/CodeUp-Llama-2-13B-Chat-HF-GPTQ`.

## Quickstart

### Python (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TheBloke/CodeUp-Llama-2-13B-Chat-HF-GPTQ"
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
ollama run codeup-llama-2-13b-chat-hf-gptq
```

> If that doesn’t work, use the Transformers path above or see `deploy/quickstarts/ollama.md` for the general workflow.

## When to use this model


- Code generation and code assistance workloads.

- You need lower latency and lower cost than a large model.


## What to watch out for


- License is unknown in metadata; verify the model card before production use.

- Treat upstream metadata as hints; validate behavior on your own prompts and data.

- If you see quality regressions, pin a model revision instead of tracking `main`.

- Quantization can change behavior; re-run acceptance tests after changing dtype/bit-width.


## Metadata (from Hugging Face)

| Field | Value |
|---|---|
| **Model ID** | `TheBloke/CodeUp-Llama-2-13B-Chat-HF-GPTQ` |
| **Author** | TheBloke |
| **Pipeline tag** | text-generation |
| **Library** | transformers |
| **License** | unknown |
| **Last modified** | 2023-09-27T12:45:14+00:00 |
| **Downloads** | 20 |
| **Likes** | 15 |

## Links

- [🤗 Hugging Face model page](https://huggingface.co/TheBloke/CodeUp-Llama-2-13B-Chat-HF-GPTQ)
- [Model card](https://huggingface.co/TheBloke/CodeUp-Llama-2-13B-Chat-HF-GPTQ#model-card)

---

*This page is auto-generated from Hugging Face Hub metadata. If something looks wrong, please open a PR or issue.*
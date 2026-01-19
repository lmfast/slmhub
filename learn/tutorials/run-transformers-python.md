---
title: Run with Transformers (Python)
description: Run an SLM via Hugging Face Transformers in Python.
---

# Run with Transformers (Python)

## Install

```bash
pip install transformers accelerate torch
```

## Run

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/Phi-4"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

prompt = "Write a SQL query that returns top 10 customers by revenue."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Next steps

- If you need an API endpoint: `learn/tutorials/serve-openai-compatible.md`
- If you need RAG: `learn/tutorials/rag-with-slm.md`



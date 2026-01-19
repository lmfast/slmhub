---
title: Transformers.js
description: In-browser experiments with WebGPU (when supported).
---

# Transformers.js (WebGPU)

## Why use it

- Great for demos and “try it now” experiences.
- No server needed for small-enough models.

## Minimal example (conceptual)

```javascript
import { pipeline } from "@huggingface/transformers";

const generator = await pipeline("text-generation", "onnx-community/SmolLM-360M", {
  device: "webgpu",
});

const out = await generator("Explain SLMs in one sentence.", { max_new_tokens: 32 });
console.log(out[0].generated_text);
```

## Caveats

- Browser + GPU support varies.
- Model size constraints are real; expect to stay small.



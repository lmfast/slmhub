---
title: "Resources"
description: "Essential tools, libraries, and resources for working with Small Language Models"
---

# SLM Resources

Everything you need to work with Small Language Models in 2026.

## HuggingFace Ecosystem

### Core Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| **transformers** | Load and run models | `pip install transformers` |
| **peft** | LoRA/QLoRA fine-tuning | `pip install peft` |
| **trl** | Training with SFT/RLHF | `pip install trl` |
| **datasets** | Load training data | `pip install datasets` |
| **accelerate** | Distributed training | `pip install accelerate` |

### Quick Start

```python
# Load any model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4")
```

## Inference Frameworks

```
┌─────────────────────────────────────────────────────────────────┐
│  CHOOSE YOUR TOOL                                               │
│  ────────────────                                               │
│                                                                 │
│  Local development?  ──▶  Ollama (easiest)                     │
│                                                                 │
│  Production GPU?     ──▶  vLLM (highest throughput)            │
│                                                                 │
│  CPU/Edge device?    ──▶  llama.cpp (most portable)            │
│                                                                 │
│  Browser?            ──▶  Transformers.js (no server)          │
│                                                                 │
│  NVIDIA optimized?   ──▶  TensorRT-LLM (fastest on NVIDIA)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Tool | Best For | GPU | CPU |
|------|----------|-----|-----|
| Ollama | Development | ✓ | ✓ |
| vLLM | Production | ✓ | ✗ |
| llama.cpp | Edge | ✓ | ✓ |
| Transformers.js | Browser | ✓ | ✓ |

## Vector Databases

For RAG and semantic search:

| Database | Best For | Type |
|----------|----------|------|
| ChromaDB | Development | Local |
| Qdrant | Production | Self-host/Cloud |
| Pinecone | Enterprise | Cloud |
| pgvector | PostgreSQL users | Self-host |

### ChromaDB Quick Start

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

collection.add(
    documents=["doc1", "doc2"],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["search term"],
    n_results=2
)
```

## Embedding Models

| Model | Dims | Speed | Quality |
|-------|------|-------|---------|
| all-MiniLM-L6-v2 | 384 | ⚡⚡⚡ | ⭐⭐⭐ |
| bge-base-en-v1.5 | 768 | ⚡⚡ | ⭐⭐⭐⭐ |
| bge-large-en-v1.5 | 1024 | ⚡ | ⭐⭐⭐⭐⭐ |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embedding = model.encode("Your text")
```

## Cloud GPU Access

| Platform | Cost | Best For |
|----------|------|----------|
| Google Colab | Free/Pro | Experimentation |
| RunPod | $0.20/hr | Training |
| Lambda Labs | $0.50/hr | Production |

### Colab Template

```python
# Check GPU
!nvidia-smi

# Install
!pip install -q transformers peft trl accelerate

# Load model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("...")
```

## Learning Resources

### Free Courses
- [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course)
- [Fast.ai Deep Learning](https://course.fast.ai/)

### Key Papers
- "Attention Is All You Need" - Transformers
- "LoRA" - Efficient fine-tuning
- "GPTQ" - Quantization

## Quick Links

- [Model Directory](/slmhub/docs/models/) - Browse SLMs
- [Deployment](/slmhub/docs/deploy/) - Production guides
- [Fundamentals](/slmhub/docs/learn/fundamentals/) - Core concepts

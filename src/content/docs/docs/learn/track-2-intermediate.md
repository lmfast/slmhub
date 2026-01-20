---
title: "Track 2 - Intermediate Developer (12-15 hours)"
description: "Deep dive into SLM architecture, advanced fine-tuning, RAG systems, and production deployment"
---

# Track 2: Intermediate Developer

This track teaches you to understand and build SLMs from the ground up. You'll learn architecture, advanced techniques, and production patterns.

## Prerequisites

- Completed Track 1
- Strong Python skills
- Understanding of neural networks basics
- Comfortable with PyTorch

## 2.1 Building SLMs from Scratch (4 hours)

### Why Build from Scratch?

Understanding the internals helps you:
- Debug issues effectively
- Optimize for your use case
- Contribute to open-source projects
- Build custom architectures

### Part 1: Tokenization (45 min)

**BPE Algorithm Explained**

Byte-Pair Encoding (BPE) builds a vocabulary by iteratively merging frequent pairs:

```
Corpus: "low low low lower lower lowest"

Iteration 1: ['l', 'o', 'w', 'e', 'r', ...]
Iteration 2: ['lo' (merged), 'w', 'e', 'r', ...]
Iteration 3: ['low' (merged), 'er', ...]
Iteration 4: ['lower' (merged), 'est']
```

**Build Your Own Tokenizer:**

```python
from tokenizers import Tokenizer, models, trainers

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Train on your corpus
trainer = trainers.BpeTrainer(vocab_size=10000)
tokenizer.train_from_iterator(your_corpus, trainer)

# Test
encoded = tokenizer.encode("Your test text")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
```

### Part 2: Transformer Architecture (60 min)

**Core Components:**

1. **Embedding Layer**: Converts token IDs to vectors
2. **Position Encoding**: Adds positional information
3. **Transformer Blocks**: Attention + Feed-forward
4. **Output Layer**: Converts vectors to token probabilities

**Simplified Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.W_o(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SimpleSLM(nn.Module):
    def __init__(self, vocab_size=32000, d_model=1024, n_layers=12, n_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
```

### Part 3: Training Loop (90 min)

**Complete Training Script:**

```python
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch.optim as optim

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Training setup
model = SimpleSLM(vocab_size=len(tokenizer), d_model=512, n_layers=6)
optimizer = optim.AdamW(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    
    for batch in DataLoader(tokenized_dataset, batch_size=8, collate_fn=data_collator):
        optimizer.zero_grad()
        
        outputs = model(batch["input_ids"])
        loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            batch["labels"].view(-1),
            ignore_index=-100
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
```

### Part 4: Evaluation (45 min)

**Benchmark Your Model:**

```python
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model=model,
    tasks=["hellaswag", "arc"],
    batch_size=16
)

print(results)
```

## 2.2 Advanced Fine-Tuning (3 hours)

### QLoRA vs Full Fine-Tuning vs LoRA

**Comparison:**

| Method | VRAM | Training Time | Accuracy | Cost |
|--------|------|---------------|----------|------|
| Full | 28GB | 6hr | 87.3% | $12 |
| LoRA | 12GB | 2hr | 86.9% | $4 |
| QLoRA | 7GB | 3hr | 86.5% | $6 |

**QLoRA** gives 99% of full performance at 25% VRAM!

### QLoRA Implementation

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/smolLM2-1.7B-Instruct",
    quantization_config=bnb_config
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
```

### Multi-Task Fine-Tuning

Train one model on multiple tasks:

```python
# Combine datasets
sentiment_data = load_dataset("sentiment140")
ner_data = load_dataset("conll2003")

# Multi-task training
def multi_task_collate(batch):
    # Mix batches from different tasks
    return {
        "input_ids": ...,
        "labels": ...,
        "task_id": ...  # Which task this batch belongs to
    }

# Training with task-specific heads
class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.sentiment_head = nn.Linear(base_model.config.hidden_size, 2)
        self.ner_head = nn.Linear(base_model.config.hidden_size, num_ner_labels)
    
    def forward(self, input_ids, task_id):
        outputs = self.base(input_ids)
        
        if task_id == "sentiment":
            return self.sentiment_head(outputs.last_hidden_state)
        elif task_id == "ner":
            return self.ner_head(outputs.last_hidden_state)
```

## 2.3 RAG for SLMs (3 hours)

### Build Your AI Research Assistant

**Architecture:**

```
User Question → Embedding → Vector Search → Context Retrieval → SLM Generation → Answer
```

### Step 1: Document Ingestion

```python
from docling import DocumentConverter

converter = DocumentConverter()
doc = converter.convert("research_paper.pdf")

# Extract text
text = doc.document.export_to_markdown()
```

### Step 2: Chunking Strategy

```python
def chunk_text(text, chunk_size=512, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk))
    
    return chunks

# Test different chunk sizes
for size in [256, 512, 1024]:
    chunks = chunk_text(text, chunk_size=size)
    # Evaluate retrieval accuracy
```

### Step 3: Embedding Generation

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("nomic-embed-text-v1")

# Generate embeddings
chunk_embeddings = embedder.encode(chunks, show_progress_bar=True)

# Store in vector database
import chromadb

client = chromadb.Client()
collection = client.create_collection("research_papers")

collection.add(
    embeddings=chunk_embeddings.tolist(),
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
```

### Step 4: Hybrid Search (Dense + Sparse)

```python
from rank_bm25 import BM25Okapi

# Sparse retrieval (BM25)
tokenized_corpus = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_corpus)

def hybrid_search(query, k=5):
    # Dense retrieval
    query_embedding = embedder.encode([query])[0]
    dense_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k*2
    )
    
    # Sparse retrieval
    tokenized_query = query.split()
    sparse_scores = bm25.get_scores(tokenized_query)
    sparse_indices = np.argsort(sparse_scores)[-k*2:][::-1]
    
    # Combine and rerank
    combined = merge_results(dense_results, sparse_indices)
    return rerank(combined, query)[:k]
```

### Step 5: SLM Integration

```python
def answer_question(question, context):
    prompt = f"""Based on the following context, answer the question.
If unsure, say "I don't have enough information."

Context: {context}

Question: {question}

Answer:"""
    
    response = model.generate(
        prompt,
        max_new_tokens=200,
        temperature=0.3
    )
    
    return response
```

### Step 6: Complete RAG Pipeline

```python
def rag_pipeline(question):
    # Retrieve relevant chunks
    chunks = hybrid_search(question, k=3)
    context = "\n\n".join(chunks)
    
    # Generate answer
    answer = answer_question(question, context)
    
    return {
        "answer": answer,
        "sources": chunks,
        "context": context
    }
```

## 2.4 Production Deployment (2 hours)

### Optimization Techniques

**1. Quantization**

```python
# Compare quantizations
from transformers import AutoModelForCausalLM

model_fp16 = AutoModelForCausalLM.from_pretrained("model", torch_dtype=torch.float16)
model_int8 = AutoModelForCausalLM.from_pretrained("model", load_in_8bit=True)
model_int4 = AutoModelForCausalLM.from_pretrained("model", load_in_4bit=True)

# Benchmark
benchmark(model_fp16)  # Baseline
benchmark(model_int8)  # 2x smaller, <3% accuracy loss
benchmark(model_int4)  # 4x smaller, ~5% accuracy loss
```

**2. KV Cache Optimization**

```python
# Enable KV cache
outputs = model.generate(
    input_ids,
    use_cache=True,  # Reuse past computations
    max_new_tokens=100
)

# Static KV cache for batch processing
past_key_values = None
for token in input_tokens:
    outputs = model(token, past_key_values=past_key_values)
    past_key_values = outputs.past_key_values
```

**3. Continuous Batching (vLLM)**

```python
from vllm import LLM, SamplingParams

llm = LLM(model="HuggingFaceTB/smolLM2-1.7B-Instruct")

# Batch multiple requests
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
prompts = ["Hello", "How are you?", "Explain AI"]

outputs = llm.generate(prompts, sampling_params)

# vLLM automatically batches requests as they arrive
```

### Monitoring Setup

```python
# Track metrics
metrics = {
#     "requests_per_second": ...,
#     "latency_p50": ...,
#     "latency_p95": ...,
#     "tokens_per_second": ...,
#     "gpu_memory_used": ...,
#     "error_rate": ...
# }

# Send to Prometheus/Grafana
```

## Next Steps

- **Track 3**: Distributed training, advanced optimization, edge deployment
- **Production Patterns**: Multi-expert systems, hybrid routing
- **Advanced Concepts**: Custom architectures, research frontiers

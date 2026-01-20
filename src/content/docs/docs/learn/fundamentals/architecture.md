---
title: "Model Architecture"
description: "Understanding how SLMs are built - from transformers to attention mechanisms"
---

Understanding how Small Language Models work helps you choose the right model, optimize inference, and fine-tune effectively.

## The Big Picture

Every modern SLM follows this pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│  "Hello, how are"                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                            │
│  │   TOKENIZER     │  Text → Token IDs                          │
│  └────────┬────────┘                                            │
│           │ [15496, 11, 703, 527]                               │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   EMBEDDINGS    │  Token IDs → Vectors                       │
│  └────────┬────────┘                                            │
│           │ 4 vectors of size 4096                              │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  TRANSFORMER    │  ×32 layers                                │
│  │     BLOCKS      │  Each: Attention + FFN                     │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   OUTPUT HEAD   │  Vector → Next Token Probabilities         │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│      "you" (predicted next token)                               │
└─────────────────────────────────────────────────────────────────┘
```

## Self-Attention: The Core Innovation

Attention answers: **"Which words should I pay attention to?"**

```
INPUT: "The cat sat on the mat because it was tired"

When processing "it", attention asks:
  - What does "it" refer to?
  - Look at all previous words
  - Highest attention → "cat" (0.7)
  - Medium attention → "mat" (0.2)
  - Low attention → "the", "on" (0.1)

RESULT: Model understands "it" = "cat"
```

### How Attention Works (Step by Step)

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Create Q, K, V vectors for each token                 │
│  ────────────────────────────────────────────────────────────── │
│                                                                 │
│  Token: "cat"                                                   │
│   │                                                             │
│   ├─▶ Query (Q):  "What am I looking for?"                     │
│   │               [0.2, -0.5, 0.8, ...]                        │
│   │                                                             │
│   ├─▶ Key (K):    "What information do I have?"                │
│   │               [0.1, 0.3, -0.2, ...]                        │
│   │                                                             │
│   └─▶ Value (V):  "What's my actual content?"                  │
│                   [0.9, 0.1, 0.4, ...]                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Compute attention scores                               │
│  ────────────────────────────────────────────────────────────── │
│                                                                 │
│  For token "it" looking at all tokens:                         │
│                                                                 │
│  Score = Q("it") · K(each token)                               │
│                                                                 │
│  "The"  →  0.1  (low relevance)                                │
│  "cat"  →  0.9  (high relevance!)    ← "it" refers to "cat"   │
│  "sat"  →  0.2                                                  │
│  "on"   →  0.1                                                  │
│  "the"  →  0.1                                                  │
│  "mat"  →  0.3                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Softmax → Attention weights (sum to 1.0)              │
│  ────────────────────────────────────────────────────────────── │
│                                                                 │
│  "The"  →  0.05                                                 │
│  "cat"  →  0.50  ████████████████                              │
│  "sat"  →  0.10  ███                                           │
│  "on"   →  0.05                                                 │
│  "the"  →  0.05                                                 │
│  "mat"  →  0.25  ████████                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Weighted sum of Values                                 │
│  ────────────────────────────────────────────────────────────── │
│                                                                 │
│  Output = 0.05×V("The") + 0.50×V("cat") + 0.10×V("sat") + ...  │
│                                                                 │
│  Result: New vector for "it" that contains "cat" information   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Head Attention

Instead of one attention, run **multiple in parallel** (heads):

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT                                                          │
│    │                                                            │
│    ├────────┬────────┬────────┬────────┐                       │
│    ▼        ▼        ▼        ▼        ▼                       │
│  Head 1  Head 2   Head 3   Head 4  ... Head 32                 │
│    │        │        │        │        │                       │
│    │     syntax   semantic  coreference  ...                   │
│    │     patterns  meaning  resolution                         │
│    │        │        │        │        │                       │
│    └────────┴────────┴────────┴────────┘                       │
│                      │                                          │
│                      ▼                                          │
│               CONCATENATE + PROJECT                             │
│                      │                                          │
│                      ▼                                          │
│                   OUTPUT                                        │
└─────────────────────────────────────────────────────────────────┘

Each head can learn different patterns:
- Head 1: Subject-verb agreement
- Head 2: Adjective-noun relationships
- Head 3: Long-range dependencies
```

## KV Cache: Why Generation Gets Faster

During text generation, **Key and Value vectors don't change** for previous tokens:

```
┌─────────────────────────────────────────────────────────────────┐
│  WITHOUT KV CACHE (Slow)                                        │
│  ────────────────────────                                       │
│                                                                 │
│  Step 1: Generate "Hello"                                       │
│          Compute K,V for: [Hello]                               │
│                                                                 │
│  Step 2: Generate "world"                                       │
│          Compute K,V for: [Hello, world]  ← Recompute Hello!   │
│                                                                 │
│  Step 3: Generate "!"                                           │
│          Compute K,V for: [Hello, world, !] ← Recompute all!   │
│                                                                 │
│  Total: O(n²) computations                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  WITH KV CACHE (Fast)                                           │
│  ─────────────────────                                          │
│                                                                 │
│  Step 1: Generate "Hello"                                       │
│          Compute K,V for: [Hello]                               │
│          Cache: {Hello: (K₁, V₁)}                               │
│                                                                 │
│  Step 2: Generate "world"                                       │
│          Load from cache: K₁, V₁                                │
│          Compute only: K₂, V₂ for "world"                       │
│          Cache: {Hello: (K₁,V₁), world: (K₂,V₂)}                │
│                                                                 │
│  Step 3: Generate "!"                                           │
│          Load: K₁,V₁,K₂,V₂                                      │
│          Compute only: K₃, V₃ for "!"                           │
│                                                                 │
│  Total: O(n) computations ✓                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Trade-off:** KV cache uses VRAM. Longer contexts = more memory.

## Attention Variants (2026)

### Grouped Query Attention (GQA)

The current standard - balances speed and quality:

```
┌─────────────────────────────────────────────────────────────────┐
│  MULTI-HEAD (Old)           GROUPED QUERY (Modern)             │
│  ─────────────────           ────────────────────               │
│                                                                 │
│  32 Query heads              32 Query heads                     │
│  32 Key heads                 8 Key heads    ← Shared!         │
│  32 Value heads               8 Value heads  ← Shared!         │
│                                                                 │
│  Memory: 100%                Memory: ~40%                       │
│                                                                 │
│  Used by: GPT-2, BERT        Used by: LLaMA, Phi, Qwen, Gemma  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Flash Attention

Not a new attention type - an **optimization** for existing attention:

```
┌─────────────────────────────────────────────────────────────────┐
│  THE PROBLEM: GPU Memory Hierarchy                              │
│  ──────────────────────────────────                             │
│                                                                 │
│  ┌─────────────────┐                                            │
│  │  GPU HBM (VRAM) │  Large (24GB+) but SLOW                   │
│  │  ████████████   │                                            │
│  └────────┬────────┘                                            │
│           │ ← Bottleneck! Data transfer is slow                │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   GPU SRAM      │  Tiny (20MB) but FAST                     │
│  │   ████          │                                            │
│  └─────────────────┘                                            │
│                                                                 │
│  Standard attention: Moves huge matrices back and forth         │
│  Flash attention: Keeps computation in fast SRAM                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  FLASH ATTENTION SOLUTION: Tiling                               │
│  ──────────────────────────────                                 │
│                                                                 │
│  Instead of:                                                    │
│    1. Load entire Q, K, V to SRAM  ← Too big!                  │
│    2. Compute attention                                         │
│    3. Write back to HBM                                         │
│                                                                 │
│  Flash Attention:                                               │
│    1. Load TILE of Q, K, V  ← Small chunk                      │
│    2. Compute partial attention                                 │
│    3. Accumulate results                                        │
│    4. Repeat for all tiles                                      │
│    5. Never store full attention matrix!                        │
│                                                                 │
│  Result: 2-4x faster, much less memory                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Mixture of Experts (MoE)

The secret to huge capacity with reasonable compute:

```
┌─────────────────────────────────────────────────────────────────┐
│  DENSE MODEL                    MoE MODEL                       │
│  ───────────                    ─────────                       │
│                                                                 │
│  Input                          Input                           │
│    │                              │                             │
│    ▼                              ▼                             │
│  ┌──────────────┐              ┌──────────┐                     │
│  │   FFN        │              │  Router  │ ← Picks experts    │
│  │  (100% used) │              └────┬─────┘                     │
│  └──────────────┘                   │                           │
│                               ┌─────┼─────┐                     │
│                               ▼     ▼     ▼                     │
│                            ┌────┐┌────┐┌────┐┌────┐···┌────┐   │
│  7B params                 │ E1 ││ E2 ││ E3 ││ E4 │   │E64 │   │
│  7B active                 └────┘└────┘└────┘└────┘   └────┘   │
│                               │     │     │                     │
│                               └─────┴─────┘                     │
│                                  Only 2 experts active!         │
│                                                                 │
│                            47B total params                     │
│                            ~7B active per token                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

BENEFITS:
✅ More knowledge stored (47B params)
✅ Same inference speed (only 7B active)
✅ Experts specialize (code, math, languages)

USED BY: Mixtral, DeepSeek, Qwen-MoE
```

## Mamba / State Space Models

Alternative to Transformers with **linear** scaling:

```
┌─────────────────────────────────────────────────────────────────┐
│  TRANSFORMER                    MAMBA (SSM)                     │
│  ───────────                    ───────────                     │
│                                                                 │
│  Attention: O(n²)               State Update: O(n)              │
│                                                                 │
│  Sequence length 4K:            Sequence length 4K:             │
│    16 million ops                 4 thousand ops                │
│                                                                 │
│  Sequence length 128K:          Sequence length 128K:           │
│    16 BILLION ops                 128 thousand ops              │
│                                                                 │
│  Best for: Most tasks           Best for: Very long contexts    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

HOW MAMBA WORKS:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Token 1 ──▶ [State] ──┬──▶ Output 1                           │
│                        │                                        │
│  Token 2 ──▶ [State] ──┼──▶ Output 2    State carries memory   │
│                  ▲     │                 through sequence       │
│                  └─────┘                                        │
│  Token 3 ──▶ [State] ────▶ Output 3                            │
│                                                                 │
│  Key insight: Selective state updates                           │
│  - Remember important info                                      │
│  - Forget irrelevant info                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

2026: Hybrid models combine Mamba + Attention (Jamba, Hymba)
```

## Model Comparison Table

| Model | Params | Layers | Attention | MoE | Context |
|-------|--------|--------|-----------|-----|---------|
| Phi-4 | 14B | 40 | GQA | No | 16K |
| Qwen3-8B | 8B | 36 | GQA | No | 128K |
| Gemma-3-4B | 4B | 34 | GQA | No | 128K |
| SmolLM3 | 3B | 32 | GQA | No | 8K |
| Mixtral-8x7B | 47B | 32 | GQA | Yes | 32K |
| DeepSeek-V3 | 671B | 61 | GQA | Yes | 128K |

## Memory Calculation

```python
def estimate_memory(params_billions, precision="fp16"):
    """Estimate model memory in GB."""
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5
    }
    return params_billions * bytes_per_param[precision]

# Examples:
# 7B FP16: 7 × 2 = 14 GB
# 7B INT4: 7 × 0.5 = 3.5 GB
```

## Key Takeaways

1. **Attention is key** - Enables understanding context
2. **KV Cache speeds generation** - Trades memory for speed
3. **GQA is standard** - Efficient key/value sharing
4. **Flash Attention optimizes memory** - Hardware-aware computation
5. **MoE scales knowledge** - More params, same compute
6. **Mamba for long contexts** - Linear scaling alternative

## Next Steps

- [Quantization](/slmhub/docs/learn/fundamentals/quantization/) - Make models smaller
- [Fine-Tuning](/slmhub/docs/learn/fundamentals/fine-tuning/) - Adapt with LoRA
- [Model Directory](/slmhub/docs/models/) - Compare specific models

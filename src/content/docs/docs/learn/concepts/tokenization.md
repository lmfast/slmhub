---
title: "Tokenization - How Models \"See\" Text"
description: "Understanding tokenization algorithms, vocabulary trade-offs, and multilingual tokenization"
---

import ColabButton from '../../../../../components/ColabButton.astro';

<ColabButton path="docs/learn/concepts/tokenization.ipynb" />

# Tokenization: How Models "See" Text

Tokenization is the bridge between human language and machine understanding. This guide covers algorithms, trade-offs, and practical implementation.

## The Fundamental Problem

Computers understand numbers. Humans understand words. Tokenization bridges the gap.

**Example:**
```
Input: "The quick brown fox"

Tokenization:
"The quick brown fox" → [464, 2068, 7586, 21831]

Embedding:
[464, 2068, 7586, 21831] → [[0.1, 0.3, ...], [0.2, -0.1, ...], ...]
```

## Tokenization Algorithms

### Algorithm 1: Word-Level (Naive)

```python
# Simplest approach
def word_tokenize(text):
    return text.split()

# Result:
word_tokenize("Hello, world!") 
# → ["Hello,", "world!"]
```

**Problems:**
- ❌ Huge vocabulary (millions of words)
- ❌ Can't handle unknown words
- ❌ Ignores word similarities (e.g., "run" vs "running")

### Algorithm 2: Character-Level

```python
def char_tokenize(text):
    return list(text)

# Result:
char_tokenize("Hello")
# → ['H', 'e', 'l', 'l', 'o']
```

**Problems:**
- ❌ Very long sequences
- ❌ Loses word meaning
- ✓ Small vocabulary (26-100 chars)

### Algorithm 3: Byte-Pair Encoding (BPE) ✓

**Used by:** GPT series, Llama, Phi

**How it works:**
1. Start with character vocabulary
2. Find most frequent character pair
3. Merge into new token
4. Repeat until vocab size reached

**Example:**
```
Corpus: "low low low lower lower lowest"

Iteration 1: 'l', 'o', 'w', 'e', 'r', ...
Iteration 2: 'lo' (frequent pair)
Iteration 3: 'low' (frequent pair)
Iteration 4: 'lower' (frequent pair)
...

Final Tokens: ["low", "er", "est"]
```

**Implementation:**

```python
# Simplified BPE implementation
def train_bpe(corpus, vocab_size=1000):
    # Start with chars
    vocab = set(char for text in corpus for char in text)
    
    # Iteratively merge
    while len(vocab) < vocab_size:
        # Count pairs
        pairs = count_pairs(corpus, vocab)
        
        # Merge most frequent
        best_pair = max(pairs, key=pairs.get)
        vocab.add(''.join(best_pair))
    
    return vocab
```

### Algorithm 4: WordPiece

**Used by:** BERT, Gemma

**Difference from BPE:** Uses likelihood instead of frequency

### Algorithm 5: SentencePiece

**Used by:** T5, Qwen, SmolLM

**Key Feature:** Language-agnostic (works without spaces)

## Vocabulary Size Trade-offs

| Vocab Size | Avg Tokens/Word | Model Params | Inference Speed |
|-----------|-----------------|--------------|----------------|
| 10,000 | 2.3 | +50M | Fast |
| 32,000 | 1.5 | +150M | Balanced |
| 100,000 | 1.1 | +470M | Slow |

**Finding:** 32K-50K is sweet spot for most SLMs

## Multilingual Tokenization

**Challenge:** English-centric tokenizers waste tokens on other languages

**Example:**
```python
# English text
tokenize("Hello world")  
# → 2 tokens

# Korean text (same meaning)
tokenize("안녕하세요 세계")
# → 12 tokens (6x more!)
```

**Solution:** Multilingual vocabularies with balanced sampling

**Example: Qwen Tokenizer**
- 150K vocabulary
- Covers 28+ languages efficiently
- Special handling for CJK characters

## Practical Lab: Build Custom Tokenizer

```python
from tokenizers import Tokenizer, models, trainers

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Train on custom corpus
trainer = trainers.BpeTrainer(vocab_size=10000)
tokenizer.train_from_iterator(your_corpus, trainer)

# Test
encoded = tokenizer.encode("Your test text")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")

# Compare to GPT-2 tokenizer
from transformers import GPT2Tokenizer
gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
comparison = compare_tokenizers(tokenizer, gpt2_tok, test_texts)
```

**Challenge:** Train tokenizer on domain-specific corpus (legal, medical, code) and measure efficiency gains

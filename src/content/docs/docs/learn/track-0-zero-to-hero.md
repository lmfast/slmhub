---
title: "Track 0 - Zero to Hero in 60 Minutes"
description: "Get from \"What's an SLM?\" to running your first model in 1 hour"
---

# Zero to Hero in 60 Minutes

Welcome! This track will take you from complete beginner to running your first Small Language Model. By the end, you'll understand what SLMs are, how they work, and have hands-on experience running one.

## What You'll Learn

- What Small Language Models are and why they matter
- How to run your first SLM (no installation required)
- Understanding the basics of how models process text
- Next steps for your learning journey

## Module 1: Understanding SLMs (15 minutes)

### What is a Small Language Model?

A Small Language Model (SLM) is an AI model that can understand and generate human-like text, but is small enough to run on your laptop, phone, or even a Raspberry Pi.

**Think of it this way:**
- **Large Language Models (LLMs)**: Like a massive library with millions of books (70B+ parameters)
- **Small Language Models (SLMs)**: Like a smart filing cabinet (1-10B parameters)
- **Both can answer questions correctly**, but SLMs are faster, cheaper, and run locally

### Why 2026 is Different

The landscape has changed dramatically:

- **2024**: LLM scaling hit diminishing returns
- **2025**: Open-source SLMs achieved parity (Qwen, Phi)
- **Jan 2026**: DeepSeek R1 shock - 1.5B beats 70B models
- **Now**: SLM-first architectures for agents

### Key Advantages of SLMs

1. **Speed**: <100ms latency on modern hardware
2. **Cost**: 90% cost reduction vs cloud LLMs
3. **Privacy**: Run entirely on your device
4. **Control**: Fine-tune for your specific needs

## Module 2: Run Your First SLM (20 minutes)

### Option A: Browser-Based (No Installation)

The easiest way to get started is running a model directly in your browser using WebGPU.

**Prerequisites:**
- Modern browser (Chrome/Edge with WebGPU support)
- 5 minutes

**Step 1: Load the Model**

We'll use Transformers.js to run a model entirely in your browser:

```javascript
import { pipeline } from '@xenova/transformers';

// Load a small model (SmolLM2-360M)
const generator = await pipeline(
  'text-generation',
  'Xenova/smolLM2-360M-instruct'
);
```

**Step 2: Generate Text**

```javascript
// Generate a haiku about AI
const result = await generator(
  'Write a haiku about artificial intelligence:',
  {
    max_new_tokens: 50,
    temperature: 0.7
  }
);

console.log(result[0].generated_text);
```

**Try it yourself:**
- Adjust the `temperature` slider (0.0 = deterministic, 1.0 = creative)
- Try different prompts
- Monitor tokens per second in the console

### Option B: Local with Ollama (Recommended)

For a more powerful experience, install Ollama and run models locally.

**Installation:**

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

**Pull Your First Model:**

```bash
# Download Phi-3.5-mini (3.8B parameters, ~2.3GB)
ollama pull phi3.5

# This will take 5-10 minutes depending on your connection
```

**Run Your First Conversation:**

```bash
ollama run phi3.5 "Explain quantum computing in simple terms"
```

**Expected Output:**
```
Quantum computing uses quantum bits (qubits) that can exist in multiple states 
simultaneously, unlike classical bits. This allows quantum computers to process 
vast amounts of information in parallel, potentially solving certain problems 
exponentially faster than classical computers.
```

### What Just Happened?

When you ran that command:
1. **Tokenization**: Your text was converted to numbers the model understands
2. **Inference**: The model processed the input and generated a response
3. **Generation**: Tokens were converted back to human-readable text

## Module 3: Understand What Just Happened (15 minutes)

### The Tokenization Process

Text doesn't go directly into the model. It's first converted to tokens (numbers).

**Example:**
```
Input: "Hello, world!"
Tokens: ['Hello', ',', ' world', '!']
Token IDs: [9906, 11, 1917, 0]
```

**Why this matters:**
- Models process tokens, not words
- Fewer tokens = faster and cheaper
- Different tokenizers handle text differently

### The Generation Process

When the model generates text, it:

1. **Reads your input** (converted to tokens)
2. **Processes through layers** (attention mechanisms)
3. **Predicts next token** (probability distribution)
4. **Samples a token** (based on temperature)
5. **Repeats** until complete

**Visualization:**
```
Input: "The capital of France is"
  ↓
[Token 1: "Paris" (95% probability)]
[Token 2: "Paris," (3% probability)]
[Token 3: "Lyon" (1% probability)]
  ↓
Output: "Paris"
```

### Attention Mechanism (Simplified)

The model "looks at" different parts of the input when generating each token:

- When generating "Paris", it focuses on "capital" and "France"
- This is called **attention** - the model learns what to pay attention to

## Module 4: Next Steps (10 minutes)

### Choose Your Learning Path

Based on your goals, pick a track:

**Track 1: Complete Beginner (4-6 hours)**
- Perfect if: You want to build practical applications
- You'll learn: Fine-tuning, deployment, RAG systems
- Time: 4-6 hours

**Track 2: Intermediate Developer (12-15 hours)**
- Perfect if: You want to understand how models work
- You'll learn: Building from scratch, advanced techniques
- Time: 12-15 hours

**Track 3: Advanced Production (20-25 hours)**
- Perfect if: You're deploying to production
- You'll learn: Optimization, distributed training, edge deployment
- Time: 20-25 hours

### Immediate Next Steps

1. **Experiment with different models:**
   ```bash
   ollama pull qwen2.5:3b
   ollama pull gemma2:2b
   ```

2. **Try different tasks:**
   - Code generation
   - Translation
   - Summarization
   - Question answering

3. **Join the community:**
   - GitHub Discussions
   - Discord server
   - Share your experiments

### Key Takeaways

✅ SLMs are powerful, fast, and run locally  
✅ You can run models in the browser or locally  
✅ Tokenization converts text to numbers  
✅ Models generate text token by token  
✅ You're ready to dive deeper!

## Practice Exercise

**Challenge:** Build a simple CLI chat app

```python
import ollama

def chat():
    print("Chat with Phi-3.5 (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        response = ollama.chat(
            model='phi3.5',
            messages=[{'role': 'user', 'content': user_input}]
        )
        
        print(f"AI: {response['message']['content']}")

if __name__ == "__main__":
    chat()
```

**Next:** Move to [Track 1: Complete Beginner](/slmhub/docs/learn/track-1-complete-beginner/) to learn fine-tuning and deployment.

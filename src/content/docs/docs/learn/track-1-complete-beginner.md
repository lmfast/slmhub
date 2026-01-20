---
title: "Track 1 - Complete Beginner (4-6 hours)"
description: "Build practical applications with SLMs - fine-tuning, deployment, and RAG systems"
---

# Track 1: Complete Beginner

This track teaches you to build real applications with Small Language Models. You'll learn fine-tuning, deployment, and building RAG systems.

## Prerequisites

- Completed Track 0 or equivalent experience
- Basic Python knowledge
- Comfortable with command line

## 1.1 Hello SLM - WebGPU in Browser (30 min)

### What You'll Build

An AI text generator running entirely in your browser - no server needed!

### Step 1: Setup

Create a simple HTML file:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Browser SLM Demo</title>
</head>
<body>
    <h1>AI Text Generator</h1>
    <textarea id="prompt" placeholder="Enter your prompt..."></textarea>
    <button onclick="generate()">Generate</button>
    <div id="output"></div>

    <script type="module">
        import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';
        
        let generator = null;
        
        async function loadModel() {
            generator = await pipeline(
                'text-generation',
                'Xenova/smolLM2-360M-instruct'
            );
            console.log('Model loaded!');
        }
        
        window.generate = async function() {
            if (!generator) {
                await loadModel();
            }
            
            const prompt = document.getElementById('prompt').value;
            const output = await generator(prompt, {
                max_new_tokens: 100,
                temperature: 0.7
            });
            
            document.getElementById('output').textContent = 
                output[0].generated_text;
        };
        
        loadModel();
    </script>
</body>
</html>
```

### Step 2: Experiment

- Try different prompts
- Adjust temperature (0.0 = deterministic, 1.0 = creative)
- Monitor performance

### What's Happening?

The model runs entirely in your browser using WebGPU, a modern API that lets web pages use your GPU for computation.

## 1.2 Local SLM with Ollama (45 min)

### Installation

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

### Pull Your First Model

```bash
# Download Phi-3.5-mini (recommended for beginners)
ollama pull phi3.5

# Expected: ~2.3GB download, 5-10 minutes
```

**While you wait:** Read about quantization below.

### Understanding Quantization

**The Problem:** Full models are large (14B params = 28GB in FP16)

**The Solution:** Quantization reduces precision:
- **FP16**: 16-bit floating point (28GB)
- **INT8**: 8-bit integer (14GB, <3% accuracy loss)
- **INT4**: 4-bit integer (7GB, ~5% accuracy loss)

**How it works:** Instead of storing weights as precise floats, we use fewer bits. The model still works, just slightly less accurate.

### First Conversation

```bash
ollama run phi3.5 "Explain machine learning in one sentence"
```

### Challenge: Build a CLI Chat App

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

## 1.3 Fine-Tuning Basics (2 hours)

### The Problem

"I want an AI that speaks like a medieval knight, but all models sound corporate."

### The Solution: Fine-Tuning

Fine-tuning teaches existing models your style or domain without retraining from scratch.

### What is LoRA?

**LoRA (Low-Rank Adaptation)** trains tiny "adapter" layers instead of all parameters:

- **Full fine-tuning**: Train all 1.7B parameters (slow, expensive)
- **LoRA**: Train <1% of parameters (fast, cheap, 99% as good)

**Visual:**
```
Original Model: 1.7B parameters
  ↓
Add LoRA adapters: +5M parameters (0.3%)
  ↓
Train only adapters
  ↓
Result: Model with your style, 99% of original capability
```

### Hands-On: Fine-Tune SmolLM2-1.7B

**Step 1: Prepare Dataset**

Create `dataset.jsonl`:

```json
{"instruction": "Write a greeting", "output": "Hark! Verily, I standeth ready to assist..."}
{"instruction": "Explain the weather", "output": "By mine troth, the skies doth show..."}
```

**Step 2: Configure LoRA**

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Load model
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/smolLM2-1.7B-Instruct")

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank (lower = fewer parameters)
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1
)

# Apply LoRA
model = get_peft_model(model, lora_config)
```

**Step 3: Train**

```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
```

**Estimated Time:** 15-20 minutes on Colab T4 GPU  
**Cost:** $0 (free tier)

**Step 4: Test Your Model**

```python
# Compare outputs
base_output = base_model.generate("Hello!")
your_output = fine_tuned_model.generate("Hello!")

print(f"Base: {base_output}")
print(f"Yours: {your_output}")
```

### Step 5: Share on Hugging Face

```python
model.push_to_hub("your-username/medieval-knight-slm")
```

## 1.4 Deploy Your First SLM (1.5 hours)

### Choose Your Deployment Target

**Cloud (FastAPI)** - Best for: Production apps, high traffic  
**Edge (Raspberry Pi)** - Best for: Privacy, offline use  
**Browser (WebGPU)** - Best for: No installation, demos

### Option A: Cloud Deployment with FastAPI

**Step 1: Create API Server**

```python
from fastapi import FastAPI
from transformers import pipeline
import uvicorn

app = FastAPI()
generator = pipeline("text-generation", "HuggingFaceTB/smolLM2-1.7B-Instruct")

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 100):
    result = generator(prompt, max_new_tokens=max_tokens)
    return {"text": result[0]["generated_text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Step 2: Deploy to Hugging Face Spaces**

1. Create a new Space
2. Upload your code
3. Add `requirements.txt`:
   ```
   fastapi
   transformers
   torch
   uvicorn
   ```
4. Deploy!

**Step 3: Test**

```bash
curl -X POST "https://your-space.hf.space/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

### Option B: Edge Deployment (Raspberry Pi)

**Step 1: Flash Raspberry Pi OS (64-bit)**

**Step 2: Install Dependencies**

```bash
sudo apt update
sudo apt install cmake build-essential
```

**Step 3: Compile llama.cpp**

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j4
```

**Step 4: Download Optimized Model**

```bash
wget https://huggingface.co/TheBloke/phi-3.5-mini-GGUF/resolve/main/phi-3.5-mini.Q4_K_M.gguf
```

**Step 5: Test Inference**

```bash
./llama-cli -m phi-3.5-mini.Q4_K_M.gguf -n 50 -p "Hello world"
```

**Performance:**
- First token: ~180ms
- Tokens/sec: 12-15
- Memory: 2.8GB
- Power: 5W

### Option C: Browser Deployment

See section 1.1 for WebGPU implementation.

## Next Steps

- **Track 2**: Learn to build models from scratch
- **RAG Tutorial**: Build a knowledge base system
- **Production Guide**: Optimize for scale

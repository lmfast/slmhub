---
title: "Track 3 - Advanced Production (20-25 hours)"
description: "Expert-level topics: distributed training, model optimization, edge deployment, and agentic architectures"
---

# Track 3: Advanced Production

Master production-grade SLM systems. Learn distributed training, advanced optimization, edge deployment, and building agentic architectures.

## Prerequisites

- Completed Track 2
- Experience with distributed systems
- Understanding of GPU architecture
- Production deployment experience

## 3.1 Distributed Training (5 hours)

### Why Distributed Training?

For models >10B parameters or large datasets, single-GPU training is impractical:
- **3B model**: Requires 4x A100 GPUs
- **7B model**: Requires 8x A100 GPUs
- **13B model**: Requires 16x A100 GPUs with ZeRO

### Part 1: Data Parallelism

**Concept:** Each GPU processes different batches, gradients are averaged.

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend="nccl")

# Wrap model
model = DDP(model, device_ids=[rank])

# Training loop (same as single GPU)
for batch in dataloader:
    outputs = model(batch)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

**Speedup:** 4x GPUs ≈ 3.5x faster (communication overhead)

### Part 2: Model Parallelism

**When:** Model doesn't fit on single GPU

```python
# Split model across GPUs
class ModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        # GPU 0: Layers 0-8
        self.layers_0_8 = nn.Sequential(...).to(0)
        # GPU 1: Layers 9-16
        self.layers_9_16 = nn.Sequential(...).to(1)
        # GPU 2: Layers 17-24
        self.layers_17_24 = nn.Sequential(...).to(2)
        # GPU 3: Layers 25-32
        self.layers_25_32 = nn.Sequential(...).to(3)
    
    def forward(self, x):
        x = self.layers_0_8(x.to(0))
        x = self.layers_9_16(x.to(1))
        x = self.layers_17_24(x.to(2))
        x = self.layers_25_32(x.to(3))
        return x
```

### Part 3: DeepSpeed ZeRO

**ZeRO-1:** Shard optimizer states  
**ZeRO-2:** + Shard gradients  
**ZeRO-3:** + Shard parameters

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    },
    "offload_param": {
      "device": "cpu"
    }
  }
}
```

**Result:** Train 13B model on 4x 24GB GPUs (impossible without ZeRO)

### Practical Project: Train 3B Model

```python
from deepspeed import initialize

# Initialize DeepSpeed
model_engine, optimizer, _, _ = initialize(
    model=model,
    optimizer=optimizer,
    config="ds_config.json"
)

# Training loop
for batch in dataloader:
    outputs = model_engine(batch)
    loss = criterion(outputs, targets)
    model_engine.backward(loss)
    model_engine.step()
```

## 3.2 Model Optimization (4 hours)

### Optimization Decision Tree

**Mobile (iOS/Android):**
- Use: TFLite + INT8 quantization
- Result: 3.5GB → 900MB, 15ms latency

**Raspberry Pi:**
- Use: llama.cpp + Q4_K_M quantization
- Result: 28GB → 2.1GB, 50ms latency

**Browser (WebGPU):**
- Use: ONNX + INT4 quantization
- Result: 28GB → 1.8GB, 80ms latency

### Quantization Methods Compared

| Method | Size | MMLU | Latency | Notes |
|--------|------|------|---------|-------|
| FP16 | 28GB | 84.8 | 45ms | Baseline |
| INT8 | 14GB | 84.3 | 32ms | Best balance |
| INT4 | 7GB | 82.1 | 28ms | Some accuracy loss |
| GPTQ-4 | 7.2GB | 83.8 | 30ms | Better than naive INT4 |
| AWQ-4 | 7.1GB | 84.0 | 29ms | Best INT4 method |

### GPTQ Quantization

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Quantize model
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

model = AutoGPTQForCausalLM.from_pretrained(
    "HuggingFaceTB/smolLM2-1.7B-Instruct",
    quantize_config=quantize_config
)

# Calibrate on sample data
model.quantize(calibration_dataset)

# Save quantized model
model.save_quantized("smolLM2-1.7B-gptq-4bit")
```

### Pruning

**Structured Pruning:** Remove entire neurons  
**Unstructured Pruning:** Remove individual weights

```python
import torch.nn.utils.prune as prune

# Prune 30% of weights
prune.l1_unstructured(
    module,
    name="weight",
    amount=0.3
)

# Remove pruned weights permanently
prune.remove(module, "weight")
```

### Practical Pipeline: Optimize for Raspberry Pi

```bash
# Step 1: Convert to GGUF
python convert.py phi-4 --outtype q4_k_m

# Step 2: Test on sample hardware
./llama-cli -m phi-4-q4_k_m.gguf -n 100 -p "Explain quantum computing"

# Step 3: Benchmark
# Latency: 52ms avg
# Memory: 2.3GB peak
# Accuracy: 83.1% MMLU (vs 84.8% original)
```

## 3.3 Edge Deployment (5 hours)

### Raspberry Pi 5 Setup

**Hardware:** 8GB RAM, ARM Cortex-A76

**Step 1: Flash Raspberry Pi OS (64-bit)**

**Step 2: Install Dependencies**

```bash
sudo apt update
sudo apt install cmake build-essential python3-pip
```

**Step 3: Compile llama.cpp for ARM**

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j4  # Use all 4 cores
```

**Step 4: Download Optimized Model**

```bash
wget https://huggingface.co/TheBloke/phi-3.5-mini-GGUF/resolve/main/phi-3.5-mini.Q4_K_M.gguf
```

**Step 5: Create API Server**

```python
from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['prompt']
    result = subprocess.run(
        ['./llama-cli', '-m', 'phi-3.5-mini.Q4_K_M.gguf', 
         '-p', prompt, '-n', '100'],
        capture_output=True
    )
    return {'text': result.stdout.decode()}

app.run(host='0.0.0.0', port=5000)
```

**Performance:**
- First token: 180ms
- Tokens/sec: 12-15
- Memory: 2.8GB
- Power: 5W

### Mobile Deployment

**iOS (Core ML):**

```python
import coremltools as ct

mlmodel = ct.convert(
    model,
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL  # Use Neural Engine
)
mlmodel.save("phi-mini.mlpackage")
```

**Android (TensorFlow Lite):**

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

### Real-World Project: Smart Home Assistant

**Components:**
- Whisper-tiny (speech → text)
- Phi-3.5-mini-Q4 (text → intent)
- Custom TTS (text → speech)

**Performance:**
- Wake word detection: <50ms
- Speech recognition: ~2 seconds
- AI response generation: ~5 seconds
- Total latency: <10 seconds

**Cost:** ~$80 (Pi + mic + speaker) vs $15/month cloud

## 3.4 Agentic Architectures (6 hours)

### Why SLMs for Agents?

| Aspect | LLM Agent | SLM Agent |
|--------|-----------|-----------|
| Latency | 500-2000ms | 50-200ms |
| Cost/call | $0.002 | $0.0001 |
| Modularity | Monolithic | Composable |
| Offline | ❌ | ✓ |

**Key Insight:** Agents make MANY calls → SLM savings compound

### Multi-Expert System Architecture

```
User Query
    ↓
┌─────────────┐
│   Router    │ ← SmolLM2-1.7B (fast classification)
│     SLM     │
└──────┬──────┘
       │
       ├─────→ Code Task → CodeGemma-7B → Execute
       ├─────→ Math Task → Qwen2.5-Math → Solve
       ├─────→ General → Phi-4 → Respond
       └─────→ Vision → Gemma-3n-E2B → Analyze
```

### Build Customer Support Agent

**Step 1: Tool Calling Setup**

```python
tools = [
    {
        "name": "search_kb",
        "description": "Search knowledge base for answers",
        "parameters": {"query": "string"}
    },
    {
        "name": "create_ticket",
        "description": "Create support ticket",
        "parameters": {"issue": "string", "priority": "enum"}
    }
]

system_prompt = """You are a customer support agent.
Use available tools to help customers.
Response format:
<tool>tool_name</tool>
<params>{"param": "value"}</params>
"""
```

**Step 2: Execution Loop**

```python
def run_agent(user_message):
    conversation = [{"role": "user", "content": user_message}]
    
    while True:
        response = model.generate(
            conversation + [{"role": "system", "content": system_prompt}]
        )
        
        if "<tool>" in response:
            tool, params = parse_tool_call(response)
            result = execute_tool(tool, params)
            conversation.append({"role": "tool", "content": result})
        else:
            return response
```

**Step 3: Multi-Agent Coordination**

```python
# Agent 1: Triage (SmolLM2-1.7B)
def triage_agent(query):
    category = classify(query)  # "billing", "technical", "general"
    priority = assess_priority(query)  # "low", "medium", "high"
    return category, priority

# Agent 2: Specialist (task-specific SLM)
def specialist_agent(category, query):
    if category == "billing":
        return billing_expert(query)  # Fine-tuned on billing data
    elif category == "technical":
        return tech_expert(query)
    else:
        return general_expert(query)

# Orchestration
category, priority = triage_agent(user_query)
answer = specialist_agent(category, user_query)

if priority == "high" and confidence(answer) < 0.8:
    escalate_to_human(user_query, answer)
else:
    return answer
```

### Self-Improving Agents

**Concept:** Agent learns from user feedback

```python
# After each interaction
if user_feedback == "helpful":
    add_to_positive_examples(query, response)
elif user_feedback == "unhelpful":
    add_to_negative_examples(query, response)
    # Retrain on updated dataset
    fine_tune_model(updated_dataset)
```

**Result:** Customer support bot improved from 72% → 89% satisfaction over 3 months

## Next Steps

- **Research Frontiers:** World models, multimodal SLMs
- **Production Patterns:** Advanced routing, hybrid systems
- **Contributing:** Open-source projects, model development

#!/usr/bin/env python3
"""Generate Section 2 (Models) and Section 3 (Hands-On) notebooks"""

import json
import os

def create_cell(cell_type, source, execution_count=None):
    cell = {"cell_type": cell_type, "metadata": {}, "source": source if isinstance(source, list) else [source]}
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = []
    return cell

def create_notebook(title, goal, time, concepts, cells_content):
    cells = [
        create_cell("markdown", f"# {title}\n\n**Goal**: {goal}\n\n**Time**: {time}\n\n**Concepts Covered**:\n" + "\n".join(f"- {c}" for c in concepts)),
        create_cell("markdown", "## Setup"),
        create_cell("code", "!pip install torch transformers accelerate -q"),
    ]
    cells.extend(cells_content)
    cells.append(create_cell("markdown", "## Key Takeaways\n\n✅ **Module Complete**\n\n## Next Steps\n\nContinue to the next module in the course."))
    
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

# Section 2: Models
section2_notebooks = {
    "models/01_model_zoo.ipynb": {
        "title": "Module 2.1: Model Zoo",
        "goal": "Explore and compare featured SLM models",
        "time": "60 minutes",
        "concepts": [
            "SmolLM family (135M, 360M, 1.7B)",
            "Phi-3 family",
            "Qwen2.5 family",
            "Gemma-2, MiniCPM, StableLM-2, TinyLlama",
            "Interactive comparison tool",
            "Memory requirements per model"
        ],
        "cells": [
            create_cell("code", """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model zoo
models = {
    "SmolLM-135M": "HuggingFaceTB/SmolLM2-135M",
    "SmolLM-360M": "HuggingFaceTB/SmolLM2-360M",
    "SmolLM-1.7B": "HuggingFaceTB/SmolLM2-1.7B",
    "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

print("Available Models:")
for name, path in models.items():
    print(f"  {name}: {path}")"""),
        ]
    },
    "models/02_benchmarking.ipynb": {
        "title": "Module 2.2: Benchmarking SLMs",
        "goal": "Evaluate models on standard benchmarks",
        "time": "90 minutes",
        "concepts": [
            "Perplexity",
            "MMLU (57 subjects)",
            "HumanEval (code generation)",
            "GSM8K (math reasoning)",
            "HellaSwag (commonsense)",
            "Visualize results (radar charts)"
        ],
        "cells": [
            create_cell("code", """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(model, tokenizer, text):
    \"\"\"Calculate perplexity on text\"\"\"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

print("Perplexity measures how well a model predicts text!")"""),
        ]
    },
    "models/03_domain_specific.ipynb": {
        "title": "Module 2.3: Domain-Specific Models",
        "goal": "Explore specialized SLMs for code, math, and multilingual tasks",
        "time": "50 minutes",
        "concepts": [
            "Code models: StarCoder2, CodeQwen, DeepSeek-Coder",
            "Math models: DeepSeek-Math, MathGPT",
            "Multilingual: XGLM, mGPT",
            "Build example: code completion API"
        ],
        "cells": [
            create_cell("code", """# Domain-specific models
code_models = {
    "StarCoder2": "bigcode/starcoder2-3b",
    "CodeQwen": "Qwen/CodeQwen1.5-7B",
    "DeepSeek-Coder": "deepseek-ai/deepseek-coder-1.3b",
}

math_models = {
    "DeepSeek-Math": "deepseek-ai/deepseek-math-7b",
}

multilingual_models = {
    "XGLM": "facebook/xglm-564M",
    "mGPT": "ai-forever/mGPT",
}

print("Domain-Specific Models:")
print(f"Code: {list(code_models.keys())}")
print(f"Math: {list(math_models.keys())}")
print(f"Multilingual: {list(multilingual_models.keys())}")"""),
        ]
    },
    "models/04_hardware_requirements.ipynb": {
        "title": "Module 2.4: Hardware Requirements Matrix",
        "goal": "Calculate memory and compute requirements for deployment",
        "time": "30 minutes",
        "concepts": [
            "Memory requirements matrix (FP16, INT8, INT4)",
            "Batch size calculations",
            "Interactive calculator",
            "Deployment recommendations"
        ],
        "cells": [
            create_cell("code", """def calculate_deployment_memory(model_params, precision="fp16", batch_size=1, seq_len=2048):
    \"\"\"Calculate memory needed for inference\"\"\"
    bytes_per_param = {"fp16": 2, "int8": 1, "int4": 0.5}[precision]
    model_memory = model_params * bytes_per_param / 1e9
    
    # KV cache memory (approximate)
    kv_memory = model_params * 2 * batch_size * seq_len * bytes_per_param / 1e9
    
    return model_memory + kv_memory

print("Deployment Memory Calculator:")
for model in ["135M", "360M", "1.7B", "3B", "7B"]:
    params = float(model.replace("M", "").replace("B", "")) * (1e9 if "B" in model else 1e6)
    for prec in ["fp16", "int8", "int4"]:
        mem = calculate_deployment_memory(params, prec)
        print(f"{model} ({prec}): {mem:.2f} GB")"""),
        ]
    },
}

# Section 3: Hands-On (remaining)
section3_notebooks = {
    "hands_on/03_dpo_alignment.ipynb": {
        "title": "Module 3.3: DPO (Direct Preference Optimization)",
        "goal": "Align models to human preferences using DPO",
        "time": "60 minutes",
        "concepts": [
            "DPO implementation",
            "Load Anthropic HH-RLHF dataset",
            "Align model to preferences",
            "KL penalty tuning"
        ],
        "cells": [
            create_cell("code", """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

print("DPO trains models using preference pairs (chosen vs rejected)!")"""),
        ]
    },
    "hands_on/06_inference_optimization.ipynb": {
        "title": "Module 3.6: Inference Optimization",
        "goal": "Optimize inference speed and memory",
        "time": "75 minutes",
        "concepts": [
            "Speculative decoding (draft + verifier)",
            "FlashAttention implementation",
            "Kernel optimization",
            "Speed benchmarks"
        ],
        "cells": [
            create_cell("code", """import torch
import time

def benchmark_inference(model, tokenizer, prompt, num_tokens=100):
    \"\"\"Benchmark generation speed\"\"\"
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=num_tokens)
    elapsed = time.time() - start
    tokens_per_sec = num_tokens / elapsed
    return tokens_per_sec

print("Inference optimization can provide 2-10x speedup!")"""),
        ]
    },
    "hands_on/07_deployment.ipynb": {
        "title": "Module 3.7: Deployment Strategies",
        "goal": "Deploy SLMs in production environments",
        "time": "150 minutes",
        "concepts": [
            "vLLM serving setup",
            "Mobile deployment (ONNX Runtime)",
            "Browser deployment (WebGPU with Transformers.js)",
            "Performance comparisons"
        ],
        "cells": [
            create_cell("code", """# Deployment options
deployment_methods = {
    "vLLM": "High-throughput serving (1000+ tokens/sec)",
    "ONNX Runtime": "Mobile/edge deployment",
    "Transformers.js": "Browser deployment with WebGPU",
    "Ollama": "Local deployment",
}

print("Deployment Methods:")
for method, desc in deployment_methods.items():
    print(f"  {method}: {desc}")"""),
        ]
    },
    "hands_on/09_evaluation_monitoring.ipynb": {
        "title": "Module 3.9: Evaluation & Monitoring",
        "goal": "Monitor production SLM deployments",
        "time": "40 minutes",
        "concepts": [
            "Production metrics (latency, throughput, error rate)",
            "Grafana + Prometheus setup",
            "A/B testing framework",
            "Weights & Biases integration"
        ],
        "cells": [
            create_cell("code", """# Production metrics
metrics = {
    "latency": "Time per request (ms)",
    "throughput": "Requests per second",
    "error_rate": "Percentage of failed requests",
    "token_per_sec": "Generation speed",
}

print("Key Production Metrics:")
for metric, desc in metrics.items():
    print(f"  {metric}: {desc}")"""),
        ]
    },
}

# Create all notebooks
os.makedirs("notebooks/models", exist_ok=True)
os.makedirs("notebooks/hands_on", exist_ok=True)

all_notebooks = {**section2_notebooks, **section3_notebooks}

for path, config in all_notebooks.items():
    notebook = create_notebook(
        config["title"],
        config["goal"],
        config["time"],
        config["concepts"],
        config["cells"]
    )
    full_path = f"notebooks/{path}"
    with open(full_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Created {full_path}")

print(f"\nCreated {len(all_notebooks)} notebooks!")

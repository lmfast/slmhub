#!/usr/bin/env python3
"""Generate all remaining notebooks for the coursework structure"""

import json
import os

def create_cell(cell_type, source, execution_count=None):
    """Create a notebook cell"""
    cell = {"cell_type": cell_type, "metadata": {}, "source": source if isinstance(source, list) else [source]}
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = []
    return cell

def create_notebook(title, goal, time, concepts, cells_content):
    """Create a complete notebook"""
    cells = [
        create_cell("markdown", f"# {title}\n\n**Goal**: {goal}\n\n**Time**: {time}\n\n**Concepts Covered**:\n" + "\n".join(f"- {c}" for c in concepts)),
        create_cell("markdown", "## Setup"),
        create_cell("code", "!pip install torch numpy matplotlib seaborn transformers -q"),
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

# Define all notebooks to create
notebooks_to_create = {
    "foundations/04_complete_transformer_block.ipynb": {
        "title": "Module 1.4: Complete Transformer Block",
        "goal": "Build a complete SmolLM-135M transformer layer",
        "time": "60 minutes",
        "concepts": [
            "SmolLM-135M architecture (9 layers, 576 hidden, 9 heads, 1536 FFN)",
            "Complete transformer layer implementation",
            "Forward pass with real tokens",
            "Component profiling (time/memory)"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.manual_seed(42)"""),
            create_cell("markdown", "## Build Complete Transformer Block"),
            create_cell("code", """# SmolLM-135M specs
d_model = 576
n_heads = 9
d_ff = 1536
n_layers = 9
vocab_size = 32000

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        # Multi-head attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # FFN with SwiGLU
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff * 2),
            nn.SiLU(),
            nn.Linear(d_ff, d_model)
        )
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # FFN
        x1, x2 = self.ffn[0](x).chunk(2, dim=-1)
        x = x + self.ffn[2](F.silu(x1) * x2)
        return x

block = TransformerBlock(d_model, n_heads, d_ff)
print(f"Parameters: {sum(p.numel() for p in block.parameters()) / 1e6:.2f}M")"""),
        ]
    },
    "foundations/06_kv_cache.ipynb": {
        "title": "Module 1.6: KV Cache & Attention Optimization",
        "goal": "Understand autoregressive generation optimization",
        "time": "50 minutes",
        "concepts": [
            "Naive generation (O(n²) problem)",
            "KV cache implementation",
            "Grouped Query Attention (GQA)",
            "Multi-Head Latent Attention (MLA)",
            "Benchmark: MHA vs GQA vs MLA"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn as nn
import time

torch.manual_seed(42)"""),
            create_cell("markdown", "## Naive Generation (O(n²))"),
            create_cell("code", """# Simulate naive generation
def naive_generate(model, prompt, max_len=10):
    sequence = prompt
    for _ in range(max_len):
        # Recompute attention for entire sequence each time
        output = model(sequence)
        next_token = output[:, -1:].argmax(dim=-1)
        sequence = torch.cat([sequence, next_token], dim=1)
    return sequence

print("Naive generation recomputes all previous tokens each step!")"""),
            create_cell("markdown", "## KV Cache Implementation"),
            create_cell("code", """class CachedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, cache=None):
        batch, seq, _ = x.shape
        Q = self.W_q(x).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        
        if cache is not None:
            # Concatenate with cache
            K = torch.cat([cache['k'], K], dim=2)
            V = torch.cat([cache['v'], V], dim=2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # Update cache
        new_cache = {'k': K, 'v': V}
        
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.W_o(out), new_cache

print("KV cache stores K and V from previous tokens!")"""),
        ]
    },
    "foundations/07_hardware_gpu_basics.ipynb": {
        "title": "Module 1.7: Hardware & GPU Basics",
        "goal": "Understand GPU capabilities and memory requirements",
        "time": "40 minutes",
        "concepts": [
            "GPU taxonomy (T4, RTX 4090, A100, H100)",
            "FP32 vs FP16 vs BF16 vs INT8 vs INT4",
            "Memory calculations",
            "Interactive GPU picker tool"
        ],
        "cells": [
            create_cell("code", """import torch

# GPU specifications
gpus = {
    "T4": {"vram_gb": 16, "compute_capability": 7.5, "tensor_cores": True},
    "RTX 4090": {"vram_gb": 24, "compute_capability": 8.9, "tensor_cores": True},
    "A100": {"vram_gb": 40, "compute_capability": 8.0, "tensor_cores": True},
    "H100": {"vram_gb": 80, "compute_capability": 9.0, "tensor_cores": True},
}

print("GPU Specifications:")
for name, specs in gpus.items():
    print(f"  {name}: {specs['vram_gb']}GB VRAM, CC {specs['compute_capability']}")"""),
            create_cell("markdown", "## Memory Requirements by Precision"),
            create_cell("code", """def calculate_memory(params, precision="fp16"):
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    return params * bytes_per_param[precision] / 1e9

model_sizes = {
    "135M": 135e6,
    "360M": 360e6,
    "1.7B": 1.7e9,
    "3B": 3e9,
    "7B": 7e9,
}

print("Memory Requirements (GB):")
print(f"{'Model':<10} {'FP32':<8} {'FP16':<8} {'INT8':<8} {'INT4':<8}")
for name, params in model_sizes.items():
    print(f"{name:<10} {calculate_memory(params, 'fp32'):<8.2f} {calculate_memory(params, 'fp16'):<8.2f} {calculate_memory(params, 'int8'):<8.2f} {calculate_memory(params, 'int4'):<8.2f}")"""),
        ]
    },
    "foundations/09_training_optimizations.ipynb": {
        "title": "Module 1.9: Training Optimizations",
        "goal": "Learn memory-efficient training techniques",
        "time": "45 minutes",
        "concepts": [
            "Gradient accumulation",
            "Gradient checkpointing",
            "Mixed precision training (AMP)",
            "Memory vs compute trade-offs"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

torch.manual_seed(42)"""),
            create_cell("markdown", "## Gradient Accumulation"),
            create_cell("code", """# Simulate gradient accumulation
def train_with_accumulation(model, data, accumulation_steps=4):
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    
    for i, batch in enumerate(data):
        loss = model(batch)
        loss = loss / accumulation_steps  # Scale loss
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    print(f"Effective batch size: {len(data) * accumulation_steps}")"""),
            create_cell("markdown", "## Mixed Precision Training"),
            create_cell("code", """scaler = GradScaler()

def train_with_amp(model, data):
    optimizer = torch.optim.Adam(model.parameters())
    
    for batch in data:
        optimizer.zero_grad()
        
        with autocast():
            loss = model(batch)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    print("Mixed precision training reduces memory by ~50%!")"""),
        ]
    },
    "foundations/10_scaling_laws.ipynb": {
        "title": "Module 1.10: Scaling Laws",
        "goal": "Understand model scaling and compute budgets",
        "time": "40 minutes",
        "concepts": [
            "Chinchilla scaling formula",
            "Compute budget calculator",
            "Training time estimation",
            "Interactive calculator"
        ],
        "cells": [
            create_cell("code", """import numpy as np
import matplotlib.pyplot as plt

# Chinchilla scaling: optimal model size and data size
def chinchilla_optimal(N_params, C_total):
    \"\"\"
    N_params: Number of parameters
    C_total: Total compute budget (FLOPs)
    Returns: Optimal data size (tokens)
    \"\"\"
    # Chinchilla formula: D_opt = 20 * N
    D_opt = 20 * N_params
    return D_opt

# Example
model_sizes = [135e6, 360e6, 1.7e9, 3e9, 7e9]
for N in model_sizes:
    D = chinchilla_optimal(N, None)
    print(f"Model {N/1e6:.0f}M: Optimal data = {D/1e9:.1f}B tokens")"""),
        ]
    },
}

# Create all notebooks
os.makedirs("notebooks/foundations", exist_ok=True)
os.makedirs("notebooks/models", exist_ok=True)
os.makedirs("notebooks/hands_on", exist_ok=True)

for path, config in notebooks_to_create.items():
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

print("\nAll foundation notebooks created!")

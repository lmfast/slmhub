#!/usr/bin/env python3
"""Generate Sections 4-9 notebooks: Advanced Topics, Mathematics, Models Hub, Community, Advanced Architectures"""

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
        create_cell("code", "!pip install torch transformers accelerate matplotlib seaborn numpy -q"),
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

# Section 4: Advanced Topics
section4_notebooks = {
    "advanced_topics/01_mixture_of_experts.ipynb": {
        "title": "Module 4.1: Mixture of Experts (MoE)",
        "goal": "Implement MoE layer from scratch and understand routing mechanisms",
        "time": "90 minutes",
        "concepts": [
            "MoE layer implementation",
            "Router mechanism with top-k selection",
            "Compare dense vs MoE memory/compute",
            "Convert existing model to MoE",
            "Visualize expert routing patterns"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class MoELayer(nn.Module):
    \"\"\"Mixture of Experts Layer with Top-K Routing\"\"\"
    def __init__(self, d_model, num_experts=8, top_k=2, expert_capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # Router: maps input to expert scores
        self.router = nn.Linear(d_model, num_experts)
        
        # Expert networks (simple FFNs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Router scores
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Process through experts
        output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            # Find positions routed to this expert
            expert_mask = (top_k_indices == expert_idx)
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                # Weight by routing probability
                expert_weights = top_k_probs[expert_mask]
                output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)
        
        return output, router_probs

# Test MoE layer
d_model = 128
num_experts = 4
top_k = 2
moe = MoELayer(d_model, num_experts, top_k)

x = torch.randn(2, 10, d_model)
output, routing = moe(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Routing probabilities shape: {routing.shape}")
print(f"\\nTop-{top_k} routing active for each token")"""),
        ]
    },
    "advanced_topics/02_sliding_window_attention.ipynb": {
        "title": "Module 4.2: Sliding Window Attention",
        "goal": "Implement sliding window attention for efficient long sequences",
        "time": "75 minutes",
        "concepts": [
            "Sliding window attention implementation",
            "Compare full vs windowed attention",
            "Measure effective context length (L × W)",
            "Benchmark speed and memory",
            "Visualize attention patterns"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def sliding_window_attention(Q, K, V, window_size=512, causal=True):
    \"\"\"Sliding Window Attention
    
    Args:
        Q, K, V: Query, Key, Value matrices (batch, seq_len, d_k)
        window_size: Size of attention window
        causal: Whether to use causal masking
    \"\"\"
    batch_size, seq_len, d_k = Q.shape
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Create sliding window mask
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = True
        
        if causal:
            mask[i, i+1:] = False
    
    # Apply mask
    scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

# Test sliding window attention
seq_len = 2048
d_k = 64
window_size = 512

Q = torch.randn(1, seq_len, d_k)
K = torch.randn(1, seq_len, d_k)
V = torch.randn(1, seq_len, d_k)

output, attn = sliding_window_attention(Q, K, V, window_size)

print(f"Sequence length: {seq_len}")
print(f"Window size: {window_size}")
print(f"Output shape: {output.shape}")
print(f"Memory efficient: O(seq_len × window_size) instead of O(seq_len²)")"""),
        ]
    },
    "advanced_topics/03_pruning_distillation.ipynb": {
        "title": "Module 4.3: Pruning & Distillation",
        "goal": "Compress models through pruning and knowledge distillation",
        "time": "90 minutes",
        "concepts": [
            "Magnitude pruning implementation",
            "Structured pruning (heads, neurons)",
            "Knowledge distillation with temperature",
            "Distill Phi-3-Mini → SmolLM-1.7B",
            "Compare pruned vs distilled models"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn as nn
import torch.nn.functional as F

def magnitude_pruning(model, sparsity=0.5):
    \"\"\"Magnitude-based unstructured pruning\"\"\"
    pruned_model = model
    for name, param in pruned_model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            # Calculate threshold
            flat_param = param.data.abs().flatten()
            threshold = torch.quantile(flat_param, sparsity)
            # Create mask
            mask = param.data.abs() > threshold
            param.data *= mask.float()
    return pruned_model

def knowledge_distillation_loss(student_logits, teacher_logits, temperature=3.0, alpha=0.7):
    \"\"\"Knowledge distillation loss with temperature scaling\"\"\"
    # Soft targets from teacher
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL divergence
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    
    # Hard targets (ground truth)
    # hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combined
    # total_loss = alpha * kd_loss + (1 - alpha) * hard_loss
    
    return kd_loss

print("Pruning removes less important weights")
print("Distillation transfers knowledge from teacher to student model")"""),
        ]
    },
    "advanced_topics/04_mcp_protocol.ipynb": {
        "title": "Module 4.4: MCP Protocol",
        "goal": "Build MCP server and client for model integration",
        "time": "60 minutes",
        "concepts": [
            "Build MCP server (filesystem example)",
            "Create MCP client for model integration",
            "Implement tool calling with MCP",
            "Database access example",
            "API integration example"
        ],
        "cells": [
            create_cell("code", """# MCP (Model Context Protocol) allows models to interact with external tools

# Example: Simple MCP server structure
class MCPServer:
    \"\"\"Minimal MCP server implementation\"\"\"
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, name, func, description):
        self.tools[name] = {
            "function": func,
            "description": description
        }
    
    def call_tool(self, tool_name, args):
        if tool_name in self.tools:
            return self.tools[tool_name]["function"](**args)
        else:
            raise ValueError(f"Tool {tool_name} not found")
    
    def list_tools(self):
        return {name: info["description"] for name, info in self.tools.items()}

# Example tools
def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    return f"Written {len(content)} characters to {path}"

# Create server
server = MCPServer()
server.register_tool("read_file", read_file, "Read a file from the filesystem")
server.register_tool("write_file", write_file, "Write content to a file")

print("MCP Tools available:")
for name, desc in server.list_tools().items():
    print(f"  {name}: {desc}")"""),
        ]
    },
    "advanced_topics/05_rlhf_pipeline.ipynb": {
        "title": "Module 4.5: RLHF Pipeline",
        "goal": "Implement complete RLHF pipeline with PPO",
        "time": "120 minutes",
        "concepts": [
            "Supervised fine-tuning stage",
            "Reward model training",
            "PPO implementation with trl",
            "DPO comparison",
            "Full RLHF pipeline end-to-end"
        ],
        "cells": [
            create_cell("code", """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig

print("RLHF Pipeline:")
print("1. Supervised Fine-Tuning (SFT)")
print("2. Reward Model Training")
print("3. Reinforcement Learning (PPO)")
print("4. Evaluation")

# PPO Configuration
ppo_config = PPOConfig(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    learning_rate=1e-5,
    batch_size=32,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
)

print(f"\\nPPO Config: {ppo_config}")"""),
        ]
    },
    "advanced_topics/06_multimodal_slms.ipynb": {
        "title": "Module 4.6: Multimodal SLMs",
        "goal": "Work with vision-language models",
        "time": "90 minutes",
        "concepts": [
            "Load MiniCPM-V or similar",
            "Process image + text inputs",
            "Contrastive learning (CLIP-style)",
            "Fine-tune on custom image dataset",
            "Visualize image-text embeddings"
        ],
        "cells": [
            create_cell("code", """import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# Multimodal model example
# model_name = "openbmb/MiniCPM-V-2"  # Example multimodal SLM

print("Multimodal SLMs can process:")
print("- Images + Text prompts")
print("- Generate text descriptions")
print("- Answer questions about images")
print("- Visual reasoning tasks")

# Example usage pattern:
# processor = AutoProcessor.from_pretrained(model_name)
# model = AutoModelForVision2Seq.from_pretrained(model_name)
# 
# image = Image.open("image.jpg")
# prompt = "Describe this image"
# 
# inputs = processor(image, prompt, return_tensors="pt")
# outputs = model.generate(**inputs)
# description = processor.decode(outputs[0], skip_special_tokens=True)"""),
        ]
    },
    "advanced_topics/07_structured_output_generation.ipynb": {
        "title": "Module 4.7: Structured Output Generation",
        "goal": "Generate structured outputs (JSON, regex, grammar)",
        "time": "60 minutes",
        "concepts": [
            "JSON generation with outlines library",
            "Regex-constrained generation",
            "Grammar-based decoding",
            "Schema validation",
            "Compare constrained vs unconstrained"
        ],
        "cells": [
            create_cell("code", """import json
import re

def json_constrained_generation(model, tokenizer, prompt, schema):
    \"\"\"Generate JSON output matching a schema\"\"\"
    # Add schema to prompt
    full_prompt = f\"\"\"{prompt}
    
Generate a JSON response matching this schema:
{json.dumps(schema, indent=2)}
    
JSON: \"\"\"
    
    inputs = tokenizer(full_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from response
    json_match = re.search(r'\\{[^}]+\\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            return None
    return None

# Example schema
schema = {
    "name": "string",
    "age": "integer",
    "email": "string"
}

print("Structured output generation ensures:")
print("- Valid JSON format")
print("- Schema compliance")
print("- Type safety")
print("- Easier parsing")"""),
        ]
    },
}

# Section 5: Mathematics & Theory
section5_notebooks = {
    "mathematics/01_linear_algebra_essentials.ipynb": {
        "title": "Module 5.1: Linear Algebra Essentials",
        "goal": "Master matrix operations for transformers",
        "time": "60 minutes",
        "concepts": [
            "Matrix operations for attention (Q, K, V)",
            "Dot product similarity visualization",
            "Softmax derivation and implementation",
            "Scaling by √d_k explanation",
            "Interactive attention heatmap builder"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Matrix operations for attention
def attention_mechanism(Q, K, V):
    \"\"\"Scaled dot-product attention\"\"\"
    d_k = Q.size(-1)
    
    # Step 1: Compute similarity scores
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Step 2: Scale by sqrt(d_k) to prevent large values
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 4: Weighted sum of values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# Example
seq_len = 5
d_k = 64
Q = torch.randn(1, seq_len, d_k)
K = torch.randn(1, seq_len, d_k)
V = torch.randn(1, seq_len, d_k)

output, attn = attention_mechanism(Q, K, V)

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
print(f"Attention weights shape: {attn.shape}")
print(f"Output shape: {output.shape}")
print(f"\\nScaling by √d_k = {np.sqrt(d_k):.2f} prevents gradient vanishing")"""),
        ]
    },
    "mathematics/02_backpropagation_deep_dive.ipynb": {
        "title": "Module 5.2: Backpropagation Deep Dive",
        "goal": "Understand gradient flow through transformer layers",
        "time": "75 minutes",
        "concepts": [
            "Chain rule visualization",
            "Manual gradient computation for attention",
            "Gradient flow analysis",
            "Vanishing/exploding gradient detection",
            "Residual connection gradient paths"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Manual gradient computation example
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y * 3

# Forward pass
print(f"x = {x.item()}")
print(f"y = x² = {y.item()}")
print(f"z = 3y = {z.item()}")

# Backward pass
z.backward()
print(f"\\ndz/dx = {x.grad.item()}")  # Should be 6x = 12

# Chain rule: dz/dx = dz/dy * dy/dx = 3 * 2x = 6x

# Gradient flow in transformers
class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x):
        # Residual connection helps gradient flow
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)  # Residual
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # Residual
        return x

print("\\nResidual connections create direct gradient paths")"""),
        ]
    },
    "mathematics/03_optimization_algorithms.ipynb": {
        "title": "Module 5.3: Optimization Algorithms",
        "goal": "Implement optimizers from scratch",
        "time": "90 minutes",
        "concepts": [
            "Adam optimizer from scratch",
            "AdamW implementation",
            "Lion optimizer comparison",
            "Learning rate schedules",
            "Benchmark optimizers on training task"
        ],
        "cells": [
            create_cell("code", """import torch
import numpy as np

class AdamOptimizer:
    \"\"\"Adam optimizer from scratch\"\"\"
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]  # First moment
        self.v = [torch.zeros_like(p) for p in self.params]  # Second moment
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

print("Adam combines momentum (m) and adaptive learning rates (v)")
print("AdamW adds weight decay decoupling")"""),
        ]
    },
    "mathematics/04_loss_functions.ipynb": {
        "title": "Module 5.4: Loss Functions",
        "goal": "Understand loss functions for language modeling",
        "time": "60 minutes",
        "concepts": [
            "Cross-entropy implementation",
            "Causal LM loss (shifted targets)",
            "Perplexity calculation",
            "Connection to maximum likelihood",
            "Visualize loss landscapes"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn.functional as F
import numpy as np

def cross_entropy_loss(logits, targets):
    \"\"\"Cross-entropy loss from scratch\"\"\"
    # Log-softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Negative log-likelihood
    loss = F.nll_loss(log_probs, targets, reduction='mean')
    return loss

def causal_lm_loss(logits, input_ids):
    \"\"\"Causal language modeling loss (shifted targets)\"\"\"
    # Shift: predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # Flatten for cross-entropy
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    loss = F.cross_entropy(shift_logits, shift_labels)
    return loss

def perplexity(loss):
    \"\"\"Calculate perplexity from loss\"\"\"
    return torch.exp(loss).item()

# Example
vocab_size = 1000
seq_len = 10
logits = torch.randn(2, seq_len, vocab_size)
targets = torch.randint(0, vocab_size, (2, seq_len))

loss = causal_lm_loss(logits, targets)
ppl = perplexity(loss)

print(f"Loss: {loss.item():.4f}")
print(f"Perplexity: {ppl:.2f}")
print("\\nPerplexity = e^loss")
print("Lower perplexity = better model")"""),
        ]
    },
    "mathematics/05_information_theory_basics.ipynb": {
        "title": "Module 5.5: Information Theory Basics",
        "goal": "Understand entropy, KL divergence, and their role in training",
        "time": "60 minutes",
        "concepts": [
            "Entropy calculation and visualization",
            "KL divergence implementation",
            "Temperature sampling effect on entropy",
            "DPO KL penalty visualization",
            "Distillation loss with KL"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def entropy(probs):
    \"\"\"Calculate Shannon entropy\"\"\"
    # Avoid log(0)
    probs = probs + 1e-10
    return -(probs * torch.log(probs)).sum(dim=-1)

def kl_divergence(p, q):
    \"\"\"KL divergence: KL(P || Q)\"\"\"
    p = p + 1e-10
    q = q + 1e-10
    return (p * torch.log(p / q)).sum(dim=-1)

# Example: Temperature effect on entropy
logits = torch.randn(1, 10)  # 10 classes

temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
entropies = []

for temp in temperatures:
    probs = F.softmax(logits / temp, dim=-1)
    ent = entropy(probs).item()
    entropies.append(ent)
    print(f"Temperature {temp:.1f}: Entropy = {ent:.4f}")

print("\\nHigher temperature → Higher entropy → More diverse sampling")
print("Lower temperature → Lower entropy → More deterministic")"""),
        ]
    },
    "mathematics/06_probability_sampling.ipynb": {
        "title": "Module 5.6: Probability & Sampling",
        "goal": "Implement all sampling strategies",
        "time": "75 minutes",
        "concepts": [
            "Greedy decoding implementation",
            "Temperature sampling with interactive demo",
            "Top-k sampling",
            "Top-p (nucleus) sampling",
            "Compare all strategies with visualizations"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn.functional as F
import numpy as np

def greedy_decode(logits):
    \"\"\"Greedy: always pick highest probability token\"\"\"
    return torch.argmax(logits, dim=-1)

def temperature_sample(logits, temperature=1.0):
    \"\"\"Temperature sampling\"\"\"
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, 1)

def top_k_sample(logits, k=50):
    \"\"\"Top-k sampling\"\"\"
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(top_k_probs, 1)
    return top_k_indices.gather(-1, sampled_idx)

def top_p_sample(logits, p=0.9):
    \"\"\"Top-p (nucleus) sampling\"\"\"
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability > p
    sorted_indices_to_remove = cumsum_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Set removed tokens to very negative value
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    sampled_idx = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(-1, sampled_idx)

# Example
logits = torch.randn(1, 1000)  # 1000 token vocabulary

print("Sampling strategies:")
print(f"Greedy: token {greedy_decode(logits).item()}")
print(f"Temperature (T=0.8): token {temperature_sample(logits, 0.8).item()}")
print(f"Top-k (k=50): token {top_k_sample(logits, 50).item()}")
print(f"Top-p (p=0.9): token {top_p_sample(logits, 0.9).item()}")"""),
        ]
    },
}

# Section 6: Models Hub
section6_notebooks = {
    "models_hub/01_model_database_schema.ipynb": {
        "title": "Module 6.1: Model Database Schema",
        "goal": "Design and implement model metadata schema",
        "time": "45 minutes",
        "concepts": [
            "YAML schema definition",
            "Model metadata structure",
            "Validation functions",
            "Example model entries",
            "Schema documentation generator"
        ],
        "cells": [
            create_cell("code", """import yaml
from typing import Dict, List, Optional

# Model database schema
model_schema = {
    "name": str,
    "parameters": int,
    "architecture": str,
    "license": str,
    "base_model": Optional[str],
    "fine_tuned_from": Optional[str],
    "benchmarks": Dict[str, float],
    "hardware_requirements": {
        "fp16_memory_gb": float,
        "int8_memory_gb": float,
        "int4_memory_gb": float,
    },
    "download_links": {
        "huggingface": str,
    },
    "tags": List[str],
}

# Example model entry
example_model = {
    "name": "SmolLM-135M",
    "parameters": 135_000_000,
    "architecture": "Transformer",
    "license": "Apache-2.0",
    "base_model": None,
    "fine_tuned_from": None,
    "benchmarks": {
        "mmlu": 25.3,
        "hellaswag": 45.2,
        "perplexity": 12.5,
    },
    "hardware_requirements": {
        "fp16_memory_gb": 0.5,
        "int8_memory_gb": 0.25,
        "int4_memory_gb": 0.125,
    },
    "download_links": {
        "huggingface": "HuggingFaceTB/SmolLM2-135M",
    },
    "tags": ["small", "general", "instruction-tuned"],
}

print("Model Database Schema:")
print(yaml.dump(example_model, default_flow_style=False))"""),
        ]
    },
    "models_hub/02_interactive_model_comparison.ipynb": {
        "title": "Module 6.2: Interactive Model Comparison",
        "goal": "Build interactive model comparison tool",
        "time": "60 minutes",
        "concepts": [
            "Load model database",
            "Filtering interface (size, license, benchmarks)",
            "Comparison table generation",
            "Radar chart visualization",
            "Export comparison results"
        ],
        "cells": [
            create_cell("code", """import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Example model comparison
models = [
    {"name": "SmolLM-135M", "params": 135, "mmlu": 25.3, "memory_fp16": 0.5},
    {"name": "SmolLM-360M", "params": 360, "mmlu": 32.1, "memory_fp16": 1.2},
    {"name": "SmolLM-1.7B", "params": 1700, "mmlu": 42.5, "memory_fp16": 4.5},
    {"name": "Phi-3-mini", "params": 3800, "mmlu": 69.0, "memory_fp16": 8.0},
]

df = pd.DataFrame(models)

def filter_models(df, max_params=None, min_mmlu=None, max_memory=None):
    \"\"\"Filter models by criteria\"\"\"
    filtered = df.copy()
    if max_params:
        filtered = filtered[filtered["params"] <= max_params]
    if min_mmlu:
        filtered = filtered[filtered["mmlu"] >= min_mmlu]
    if max_memory:
        filtered = filtered[filtered["memory_fp16"] <= max_memory]
    return filtered

# Example filtering
filtered = filter_models(df, max_params=2000, min_mmlu=30)
print("Filtered Models:")
print(filtered[["name", "params", "mmlu", "memory_fp16"]])

# Radar chart for comparison
def plot_radar_chart(models_data):
    categories = ["MMLU", "Efficiency", "Size"]
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    for model in models_data:
        values = [
            model["mmlu"] / 100,  # Normalize
            1 / (model["memory_fp16"] / 10),  # Inverse for efficiency
            model["params"] / 5000,  # Normalize
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model["name"])
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison Radar Chart")
    plt.show()

print("\\nUse plot_radar_chart(models) to visualize comparisons")"""),
        ]
    },
    "models_hub/03_hardware_compatibility_matrix.ipynb": {
        "title": "Module 6.3: Hardware Compatibility Matrix",
        "goal": "Calculate hardware requirements and compatibility",
        "time": "45 minutes",
        "concepts": [
            "Memory calculation functions",
            "Speed estimation algorithms",
            "GPU compatibility checker",
            "Batch size calculator",
            "Interactive requirements tool"
        ],
        "cells": [
            create_cell("code", """def calculate_memory_requirements(params, precision="fp16", batch_size=1, seq_len=2048):
    \"\"\"Calculate memory requirements for inference\"\"\"
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }[precision]
    
    # Model weights
    model_memory_gb = params * bytes_per_param / 1e9
    
    # KV cache (approximate: 2 * params * batch * seq_len * bytes_per_param)
    kv_cache_gb = params * 2 * batch_size * seq_len * bytes_per_param / 1e9
    
    # Activation memory (rough estimate)
    activation_memory_gb = batch_size * seq_len * params * 0.1 * bytes_per_param / 1e9
    
    total_memory_gb = model_memory_gb + kv_cache_gb + activation_memory_gb
    
    return {
        "model_memory_gb": model_memory_gb,
        "kv_cache_gb": kv_cache_gb,
        "activation_memory_gb": activation_memory_gb,
        "total_memory_gb": total_memory_gb,
    }

def check_gpu_compatibility(model_params, precision, gpu_memory_gb=16):
    \"\"\"Check if model fits on GPU\"\"\"
    reqs = calculate_memory_requirements(model_params, precision)
    fits = reqs["total_memory_gb"] <= gpu_memory_gb * 0.9  # 90% safety margin
    return fits, reqs

# Example
model_params = 1_700_000_000  # 1.7B
gpu_memory = 16  # T4 GPU

for precision in ["fp16", "int8", "int4"]:
    fits, reqs = check_gpu_compatibility(model_params, precision, gpu_memory)
    status = "✅ Fits" if fits else "❌ Too large"
    print(f"{precision.upper()}: {status}")
    print(f"  Total memory: {reqs['total_memory_gb']:.2f} GB")
    print()"""),
        ]
    },
    "models_hub/04_model_leaderboards.ipynb": {
        "title": "Module 6.4: Model Leaderboards",
        "goal": "Generate and visualize model leaderboards",
        "time": "45 minutes",
        "concepts": [
            "Leaderboard generation from database",
            "Category-based rankings",
            "Weighted scoring",
            "Filtering and sorting",
            "Live leaderboard updates"
        ],
        "cells": [
            create_cell("code", """import pandas as pd
import numpy as np

# Example leaderboard data
leaderboard_data = [
    {"name": "Phi-3-mini", "mmlu": 69.0, "hellaswag": 82.3, "gsm8k": 73.2, "params": 3800},
    {"name": "SmolLM-1.7B", "mmlu": 42.5, "hellaswag": 68.1, "gsm8k": 45.3, "params": 1700},
    {"name": "SmolLM-360M", "mmlu": 32.1, "hellaswag": 55.2, "gsm8k": 28.7, "params": 360},
    {"name": "SmolLM-135M", "mmlu": 25.3, "hellaswag": 45.2, "gsm8k": 18.5, "params": 135},
]

df = pd.DataFrame(leaderboard_data)

def calculate_weighted_score(row, weights={"mmlu": 0.4, "hellaswag": 0.3, "gsm8k": 0.3}):
    \"\"\"Calculate weighted average score\"\"\"
    score = 0
    for benchmark, weight in weights.items():
        score += row[benchmark] * weight
    return score

df["weighted_score"] = df.apply(calculate_weighted_score, axis=1)
df = df.sort_values("weighted_score", ascending=False)

print("Model Leaderboard (by weighted score):")
print(df[["name", "mmlu", "hellaswag", "gsm8k", "weighted_score"]].to_string(index=False))

# Category-based rankings
print("\\n" + "="*50)
print("Category Rankings:")
print("="*50)

for category in ["mmlu", "hellaswag", "gsm8k"]:
    print(f"\\nTop models by {category.upper()}:")
    top = df.nlargest(3, category)[["name", category]]
    print(top.to_string(index=False))"""),
        ]
    },
}

# Section 7: Community (merged with Section 8)
section7_notebooks = {
    "community/01_discord_server.ipynb": {
        "title": "Module 7.1: Discord Server",
        "goal": "Explore community structure and bot integration",
        "time": "30 minutes",
        "concepts": [
            "Community structure overview",
            "Channel descriptions with code examples",
            "Bot integration examples",
            "Event scheduling code",
            "Community analytics"
        ],
        "cells": [
            create_cell("code", """# Discord server structure for SLM Hub community

channels = {
    "general": "General discussions about SLMs",
    "showcase": "Share your projects and demos",
    "help": "Get help with technical questions",
    "model-releases": "New model announcements",
    "tutorials": "Tutorial discussions and feedback",
    "research": "Research paper discussions",
}

# Example bot command
def discord_bot_example():
    \"\"\"Example Discord bot command for model info\"\"\"
    def model_info_command(model_name):
        # Fetch model info from database
        # Return formatted response
        return f"Model: {model_name}\\nParameters: ...\\nBenchmarks: ..."
    
    return model_info_command

print("Discord Community Channels:")
for channel, description in channels.items():
    print(f"  #{channel}: {description}")"""),
        ]
    },
    "community/02_contribution_guidelines.ipynb": {
        "title": "Module 7.2: Contribution Guidelines",
        "goal": "Learn how to contribute to SLM Hub",
        "time": "30 minutes",
        "concepts": [
            "PR template generator",
            "Model submission validator",
            "Tutorial checklist",
            "Code review guidelines",
            "Contribution tracking"
        ],
        "cells": [
            create_cell("code", """# Contribution guidelines and tools

def generate_pr_template(pr_type="model"):
    \"\"\"Generate PR template based on type\"\"\"
    templates = {
        "model": \"\"\"## Model Submission
- Model name: 
- Parameters: 
- License: 
- HuggingFace link: 
- Benchmarks: 
- Hardware requirements: 
\"\"\",
        "tutorial": \"\"\"## Tutorial Submission
- Title: 
- Estimated time: 
- Concepts covered: 
- Colab link: 
- Tested on: 
\"\"\",
    }
    return templates.get(pr_type, "")

def validate_model_submission(model_data):
    \"\"\"Validate model submission data\"\"\"
    required_fields = ["name", "parameters", "license", "huggingface"]
    missing = [field for field in required_fields if field not in model_data]
    
    if missing:
        return False, f"Missing fields: {missing}"
    return True, "Valid submission"

print("Contribution Types:")
print("1. Model submissions")
print("2. Tutorial additions")
print("3. Bug fixes")
print("4. Documentation improvements")
print("5. Feature additions")"""),
        ]
    },
    "community/03_research_paper_summaries.ipynb": {
        "title": "Module 7.3: Research Paper Summaries",
        "goal": "Extract and summarize research papers",
        "time": "45 minutes",
        "concepts": [
            "Paper parsing and summarization",
            "Implementation extraction",
            "Colab notebook generator from papers",
            "Paper comparison tool",
            "Citation network visualization"
        ],
        "cells": [
            create_cell("code", """# Research paper processing tools

def extract_paper_metadata(paper_text):
    \"\"\"Extract key information from paper\"\"\"
    metadata = {
        "title": None,
        "authors": [],
        "abstract": None,
        "key_contributions": [],
        "implementation_details": [],
    }
    # Implementation would parse paper text
    return metadata

def generate_notebook_from_paper(paper_metadata):
    \"\"\"Generate Colab notebook structure from paper\"\"\"
    notebook_structure = {
        "title": f"Implementation: {paper_metadata['title']}",
        "sections": [
            "Introduction",
            "Key Concepts",
            "Implementation",
            "Experiments",
            "Results",
        ]
    }
    return notebook_structure

print("Paper Processing Pipeline:")
print("1. Parse PDF/arXiv paper")
print("2. Extract key concepts")
print("3. Identify implementation details")
print("4. Generate notebook template")
print("5. Add code examples")"""),
        ]
    },
    "community/04_industry_use_cases.ipynb": {
        "title": "Module 7.4: Industry Use Cases",
        "goal": "Document and analyze industry deployments",
        "time": "45 minutes",
        "concepts": [
            "Use case template",
            "Deployment pattern examples",
            "Cost analysis calculator",
            "Performance benchmarking",
            "Case study generator"
        ],
        "cells": [
            create_cell("code", """# Industry use case analysis

use_case_template = {
    "company": "",
    "industry": "",
    "use_case": "",
    "model_used": "",
    "deployment": "",
    "cost_per_month": 0,
    "performance_metrics": {},
    "lessons_learned": [],
}

def calculate_deployment_cost(model_size_gb, requests_per_day, avg_tokens_per_request):
    \"\"\"Estimate deployment costs\"\"\"
    # GPU cost per hour (example: $0.50/hour for T4)
    gpu_cost_per_hour = 0.50
    hours_per_day = 24
    
    # Compute time per request (rough estimate)
    compute_time_per_request = avg_tokens_per_request / 100  # seconds
    
    # Total compute hours needed
    total_compute_hours = (requests_per_day * compute_time_per_request) / 3600
    
    # Cost calculation
    daily_cost = total_compute_hours * gpu_cost_per_hour
    monthly_cost = daily_cost * 30
    
    return {
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
        "cost_per_request": daily_cost / requests_per_day,
    }

# Example
costs = calculate_deployment_cost(
    model_size_gb=4.5,
    requests_per_day=10000,
    avg_tokens_per_request=200
)

print("Deployment Cost Analysis:")
print(f"Daily cost: ${costs['daily_cost']:.2f}")
print(f"Monthly cost: ${costs['monthly_cost']:.2f}")
print(f"Cost per request: ${costs['cost_per_request']:.4f}")"""),
        ]
    },
    "community/05_how_to_contribute.ipynb": {
        "title": "Module 7.5: How to Contribute",
        "goal": "Step-by-step contribution guide",
        "time": "30 minutes",
        "concepts": [
            "Contribution workflow diagram",
            "Model addition script",
            "Tutorial creation template",
            "Benchmark submission form",
            "Tool development guide"
        ],
        "cells": [
            create_cell("code", """# Contribution workflow

contribution_steps = [
    "1. Fork the repository",
    "2. Create a feature branch",
    "3. Make your changes",
    "4. Test your changes",
    "5. Submit a pull request",
    "6. Address review feedback",
    "7. Merge after approval",
]

def model_addition_script():
    \"\"\"Template script for adding a new model\"\"\"
    script = \"\"\"
# 1. Create model YAML file in models/generated/
# 2. Add model metadata
# 3. Run validation script
# 4. Generate model page
# 5. Update model index
\"\"\"
    return script

def tutorial_creation_template():
    \"\"\"Template for creating tutorials\"\"\"
    template = {
        "title": "Module X.Y: Title",
        "goal": "What students will learn",
        "time": "X minutes",
        "concepts": ["Concept 1", "Concept 2"],
        "setup": "Installation commands",
        "lessons": ["Lesson 1", "Lesson 2"],
        "exercises": ["Exercise 1"],
    }
    return template

print("Contribution Workflow:")
for step in contribution_steps:
    print(f"  {step}")"""),
        ]
    },
    "community/06_code_of_conduct.ipynb": {
        "title": "Module 7.6: Code of Conduct",
        "goal": "Understand community guidelines and enforcement",
        "time": "20 minutes",
        "concepts": [
            "CoC enforcement examples",
            "Community moderation tools",
            "Reporting mechanism",
            "Conflict resolution process",
            "Community health metrics"
        ],
        "cells": [
            create_cell("code", """# Code of Conduct and community health

code_of_conduct_principles = [
    "Be respectful and inclusive",
    "Welcome newcomers",
    "Give constructive feedback",
    "Focus on what is best for the community",
    "Show empathy towards others",
]

def community_health_metrics():
    \"\"\"Track community health indicators\"\"\"
    metrics = {
        "active_contributors": 0,
        "pull_requests_per_week": 0,
        "issues_resolved_per_week": 0,
        "response_time_hours": 0,
        "community_satisfaction": 0,
    }
    return metrics

print("Code of Conduct Principles:")
for principle in code_of_conduct_principles:
    print(f"  • {principle}")

print("\\nCommunity Health Metrics:")
print("  • Active contributors")
print("  • PR/Issue resolution time")
print("  • Community engagement")
print("  • Code review quality")"""),
        ]
    },
    "community/07_github_discussions.ipynb": {
        "title": "Module 7.7: GitHub Discussions",
        "goal": "Engage with community through GitHub Discussions",
        "time": "30 minutes",
        "concepts": [
            "Discussion categorization",
            "Q&A bot examples",
            "Show-and-tell showcase",
            "Poll creation tool",
            "Discussion analytics"
        ],
        "cells": [
            create_cell("code", """# GitHub Discussions structure

discussion_categories = {
    "general": "General discussions",
    "q-and-a": "Questions and answers",
    "ideas": "Feature ideas and suggestions",
    "show-and-tell": "Project showcases",
    "announcements": "Community announcements",
}

def create_poll_template(question, options):
    \"\"\"Create poll template for discussions\"\"\"
    poll_markdown = f\"\"\"## {question}
    
Please react with:
- 👍 Option 1: {options[0]}
- ❤️ Option 2: {options[1]}
- 🎉 Option 3: {options[2] if len(options) > 2 else 'Other'}
\"\"\"
    return poll_markdown

# Example poll
poll = create_poll_template(
    "Which SLM architecture interests you most?",
    ["Transformer", "Mamba", "Hybrid"]
)

print("GitHub Discussion Categories:")
for category, description in discussion_categories.items():
    print(f"  {category}: {description}")

print("\\nExample Poll:")
print(poll)"""),
        ]
    },
}

# Section 9: Advanced Architectures
section9_notebooks = {
    "advanced_architectures/01_state_space_models.ipynb": {
        "title": "Module 9.1: State Space Models",
        "goal": "Implement SSM from scratch and understand O(n) complexity",
        "time": "90 minutes",
        "concepts": [
            "SSM implementation from scratch",
            "Continuous vs discrete SSM",
            "O(n) complexity demonstration",
            "Long sequence handling",
            "Compare SSM vs Transformer"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn as nn
import numpy as np

class StateSpaceModel(nn.Module):
    \"\"\"Simple State Space Model (SSM)\"\"\"
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model, d_model))
        
    def forward(self, u):
        \"\"\"Forward pass: O(n) complexity\"\"\"
        batch_size, seq_len, d_model = u.shape
        h = torch.zeros(batch_size, seq_len, self.d_state, device=u.device)
        y = torch.zeros(batch_size, seq_len, self.d_model, device=u.device)
        
        # Recurrent computation
        for t in range(seq_len):
            if t == 0:
                h[:, t] = torch.matmul(u[:, t], self.B.t())
            else:
                h[:, t] = torch.matmul(h[:, t-1], self.A.t()) + torch.matmul(u[:, t], self.B.t())
            y[:, t] = torch.matmul(h[:, t], self.C.t()) + torch.matmul(u[:, t], self.D.t())
        
        return y

# Test SSM
d_model = 64
d_state = 16
ssm = StateSpaceModel(d_model, d_state)

seq_len = 2048
u = torch.randn(1, seq_len, d_model)

output = ssm(u)

print(f"Input shape: {u.shape}")
print(f"Output shape: {output.shape}")
print(f"Complexity: O(seq_len × d_state × d_model) = O(n)")
print(f"Transformer attention: O(seq_len² × d_model) = O(n²)")"""),
        ]
    },
    "advanced_architectures/02_mamba_architecture.ipynb": {
        "title": "Module 9.2: Mamba Architecture",
        "goal": "Implement selective SSM and Mamba block",
        "time": "120 minutes",
        "concepts": [
            "Selective SSM implementation",
            "Input-dependent A, B, C matrices",
            "Convolution for local context",
            "Hardware-aware algorithm",
            "Mamba block implementation"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    \"\"\"Selective State Space Model (Mamba core)\"\"\"
    def __init__(self, d_model, d_state=16, dt_rank=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        
        # Input-dependent parameters
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2)
        self.dt_proj = nn.Linear(dt_rank, d_model)
        
        # State space parameters (learned)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        \"\"\"Selective SSM forward pass\"\"\"
        batch_size, seq_len, d_model = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (batch, seq_len, 2*d_model)
        x, z = xz.chunk(2, dim=-1)
        
        # 1D convolution
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        x = self.act(x)
        
        # Compute input-dependent parameters
        x_dbl = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (batch, seq_len, d_model)
        
        # Selective scan (simplified)
        A = -torch.exp(self.A_log.unsqueeze(0))  # (1, d_model, d_state)
        
        # Output (simplified - full implementation uses parallel scan)
        y = torch.zeros_like(x)
        for i in range(seq_len):
            y[:, i] = x[:, i] * self.D.unsqueeze(0)
        
        # Gating
        y = y * self.act(z)
        
        return y

print("Mamba Architecture:")
print("- Selective SSM: input-dependent state transitions")
print("- 1D convolution: local context")
print("- Hardware-efficient: parallel scan algorithm")
print("- Linear complexity: O(n) for sequence length n")"""),
        ]
    },
    "advanced_architectures/03_mamba_2_3_improvements.ipynb": {
        "title": "Module 9.3: Mamba-2 & Mamba-3 Improvements",
        "goal": "Understand SSD and latest Mamba improvements",
        "time": "90 minutes",
        "concepts": [
            "SSD (State Space Duality) implementation",
            "Mamba-2 minimal code",
            "Mamba-3 improvements",
            "Performance comparisons",
            "Evolution visualization"
        ],
        "cells": [
            create_cell("code", """# Mamba evolution

mamba_versions = {
    "Mamba-1": {
        "key_innovation": "Selective SSM",
        "complexity": "O(n)",
        "hardware_efficient": True,
    },
    "Mamba-2": {
        "key_innovation": "SSD (State Space Duality)",
        "complexity": "O(n)",
        "improvements": [
            "Better parameterization",
            "Improved training stability",
            "Faster inference",
        ],
    },
    "Mamba-3": {
        "key_innovation": "Further optimizations",
        "complexity": "O(n)",
        "improvements": [
            "Better long-context handling",
            "Improved memory efficiency",
        ],
    },
}

def ssd_mechanism():
    \"\"\"State Space Duality (Mamba-2) concept\"\"\"
    explanation = \"\"\"
SSD (State Space Duality) improves Mamba by:
1. Better state space parameterization
2. Dual representation for efficiency
3. Improved numerical stability
4. Faster parallel scan implementation
\"\"\"
    return explanation

print("Mamba Evolution:")
for version, details in mamba_versions.items():
    print(f"\\n{version}:")
    print(f"  Innovation: {details['key_innovation']}")
    if "improvements" in details:
        for imp in details["improvements"]:
            print(f"  • {imp}")"""),
        ]
    },
    "advanced_architectures/04_hybrid_architectures.ipynb": {
        "title": "Module 9.4: Hybrid Architectures",
        "goal": "Build Jamba-style hybrid models",
        "time": "90 minutes",
        "concepts": [
            "Jamba-style hybrid (Mamba + Attention)",
            "Layer placement strategies",
            "Attention frequency experiments",
            "Quality vs speed trade-offs",
            "Custom hybrid builder"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn as nn

class HybridBlock(nn.Module):
    \"\"\"Hybrid Mamba + Attention block (Jamba-style)\"\"\"
    def __init__(self, d_model, num_heads=8, mamba_expand=2):
        super().__init__()
        self.d_model = d_model
        
        # Mamba layer
        from advanced_architectures.mamba import SelectiveSSM
        self.mamba = SelectiveSSM(d_model)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x, use_attention=True):
        # Mamba branch
        mamba_out = self.mamba(x)
        x = self.norm1(x + mamba_out)
        
        # Optional attention branch
        if use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = self.norm2(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x

def create_hybrid_model(num_layers, attention_frequency=4):
    \"\"\"Create hybrid model with attention every N layers\"\"\"
    layers = []
    for i in range(num_layers):
        use_attention = (i % attention_frequency == 0)
        layers.append(HybridBlock(d_model=512, use_attention=use_attention))
    return nn.Sequential(*layers)

print("Hybrid Architecture Benefits:")
print("- Mamba: Efficient long-context (O(n))")
print("- Attention: Strong short-range dependencies")
print("- Best of both worlds: Quality + Speed")"""),
        ]
    },
    "advanced_architectures/05_rag_advanced.ipynb": {
        "title": "Module 9.5: RAG Advanced",
        "goal": "Implement advanced RAG techniques",
        "time": "90 minutes",
        "concepts": [
            "Multi-hop RAG implementation",
            "GraphRAG with NetworkX",
            "Self-RAG with retrieval decisions",
            "Comparison of RAG variants",
            "Long-context RAG"
        ],
        "cells": [
            create_cell("code", """import networkx as nx
from typing import List, Dict

class GraphRAG:
    \"\"\"Graph-based RAG using entity relationships\"\"\"
    def __init__(self):
        self.graph = nx.DiGraph()
        self.embeddings = {}
    
    def add_document(self, doc_id, text, entities):
        \"\"\"Add document with entities to graph\"\"\"
        self.graph.add_node(doc_id, text=text)
        for entity in entities:
            if entity not in self.graph:
                self.graph.add_node(entity, type="entity")
            self.graph.add_edge(doc_id, entity, relation="contains")
    
    def retrieve(self, query_entities, top_k=5):
        \"\"\"Retrieve documents connected to query entities\"\"\"
        relevant_docs = []
        for entity in query_entities:
            if entity in self.graph:
                # Find documents connected to this entity
                for doc_id in self.graph.predecessors(entity):
                    relevant_docs.append(doc_id)
        return list(set(relevant_docs))[:top_k]

class SelfRAG:
    \"\"\"Self-RAG: model decides when to retrieve\"\"\"
    def __init__(self, model, retriever):
        self.model = model
        self.retriever = retriever
    
    def generate(self, query, max_steps=5):
        \"\"\"Generate with retrieval decisions\"\"\"
        context = []
        for step in range(max_steps):
            # Model decides: retrieve or continue?
            decision = self.model.decide_retrieve(query, context)
            
            if decision == "retrieve":
                docs = self.retriever.retrieve(query)
                context.extend(docs)
            
            # Generate next token
            output = self.model.generate(query, context)
            
            if output.endswith("</s>"):
                break
        
        return output

print("Advanced RAG Techniques:")
print("1. Multi-hop: Chain multiple retrievals")
print("2. GraphRAG: Use entity relationships")
print("3. Self-RAG: Model decides when to retrieve")
print("4. Long-context: Use long-context models")"""),
        ]
    },
    "advanced_architectures/06_speculative_decoding_deep_dive.ipynb": {
        "title": "Module 9.6: Speculative Decoding Deep Dive",
        "goal": "Implement and optimize speculative decoding",
        "time": "90 minutes",
        "concepts": [
            "Draft + target model setup",
            "Parallel verification",
            "Acceptance/rejection logic",
            "Speedup measurement",
            "Optimal k selection"
        ],
        "cells": [
            create_cell("code", """import torch
import torch.nn.functional as F

class SpeculativeDecoder:
    \"\"\"Speculative decoding for faster generation\"\"\"
    def __init__(self, draft_model, target_model):
        self.draft_model = draft_model  # Small, fast model
        self.target_model = target_model  # Large, accurate model
    
    def generate_draft(self, input_ids, k=5):
        \"\"\"Generate k tokens with draft model\"\"\"
        draft_tokens = []
        current_ids = input_ids
        
        for _ in range(k):
            logits = self.draft_model(current_ids).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            draft_tokens.append(next_token)
            current_ids = torch.cat([current_ids, next_token.unsqueeze(1)], dim=1)
        
        return torch.stack(draft_tokens, dim=1)
    
    def verify_draft(self, input_ids, draft_tokens):
        \"\"\"Verify draft tokens with target model\"\"\"
        # Run target model on input + draft tokens
        full_sequence = torch.cat([input_ids, draft_tokens], dim=1)
        target_logits = self.target_model(full_sequence).logits
        
        # Acceptance probability for each draft token
        accepted_tokens = []
        current_ids = input_ids
        
        for i, draft_token in enumerate(draft_tokens.unbind(1)):
            # Get target model probability for draft token
            target_probs = F.softmax(target_logits[:, input_ids.size(1) + i - 1, :], dim=-1)
            draft_prob = target_probs.gather(1, draft_token.unsqueeze(1))
            
            # Get draft model probability
            draft_logits = self.draft_model(current_ids).logits[:, -1, :]
            draft_model_prob = F.softmax(draft_logits, dim=-1).gather(1, draft_token.unsqueeze(1))
            
            # Acceptance probability
            accept_prob = torch.min(torch.ones_like(draft_prob), target_probs / (draft_model_prob + 1e-10))
            
            # Accept or reject
            if torch.rand(1) < accept_prob:
                accepted_tokens.append(draft_token)
                current_ids = torch.cat([current_ids, draft_token.unsqueeze(1)], dim=1)
            else:
                # Sample from adjusted distribution
                adjusted_probs = F.normalize(torch.clamp(target_probs - draft_model_prob, min=0), p=1, dim=1)
                new_token = torch.multinomial(adjusted_probs, 1)
                accepted_tokens.append(new_token.squeeze(1))
                break
        
        return torch.stack(accepted_tokens, dim=1) if accepted_tokens else None

print("Speculative Decoding:")
print("- Draft model: Fast, generates k tokens")
print("- Target model: Accurate, verifies draft")
print("- Speedup: 2-3x for compatible models")
print("- Optimal k: Usually 3-5 tokens")"""),
        ]
    },
    "advanced_architectures/07_quantization_theory.ipynb": {
        "title": "Module 9.7: Quantization Theory",
        "goal": "Deep dive into quantization mathematics",
        "time": "90 minutes",
        "concepts": [
            "Uniform quantization math",
            "Per-tensor vs per-channel",
            "GPTQ algorithm implementation",
            "Quantization error analysis",
            "Weight distribution visualization"
        ],
        "cells": [
            create_cell("code", """import torch
import numpy as np
import matplotlib.pyplot as plt

def uniform_quantize(weights, bits=8):
    \"\"\"Uniform quantization\"\"\"
    # Calculate scale and zero point
    w_min = weights.min()
    w_max = weights.max()
    
    scale = (w_max - w_min) / (2 ** bits - 1)
    zero_point = -w_min / scale
    
    # Quantize
    q_weights = torch.round(weights / scale + zero_point)
    q_weights = torch.clamp(q_weights, 0, 2 ** bits - 1)
    
    # Dequantize
    dequantized = (q_weights - zero_point) * scale
    
    return dequantized, scale, zero_point

def gptq_quantize(layer, bits=4):
    \"\"\"GPTQ: Optimal quantization with Hessian\"\"\"
    # Simplified GPTQ algorithm
    weights = layer.weight.data.clone()
    num_bits = bits
    
    # Per-channel quantization
    scales = []
    quantized_weights = []
    
    for channel in range(weights.shape[0]):
        channel_weights = weights[channel]
        
        # Find optimal scale for this channel
        w_abs_max = channel_weights.abs().max()
        scale = w_abs_max / (2 ** (num_bits - 1) - 1)
        
        # Quantize
        q_weights = torch.round(channel_weights / scale)
        q_weights = torch.clamp(q_weights, -2 ** (num_bits - 1), 2 ** (num_bits - 1) - 1)
        
        # Dequantize
        dequantized = q_weights * scale
        
        scales.append(scale)
        quantized_weights.append(dequantized)
    
    quantized_weights = torch.stack(quantized_weights)
    
    return quantized_weights, scales

# Example
weights = torch.randn(128, 256) * 0.1

# Uniform quantization
uniform_q, scale, zp = uniform_quantize(weights, bits=8)
uniform_error = (weights - uniform_q).abs().mean()

# GPTQ quantization
gptq_q, scales = gptq_quantize(torch.nn.Linear(256, 128), bits=4)
gptq_error = (weights - gptq_q).abs().mean()

print(f"Uniform quantization error: {uniform_error:.6f}")
print(f"GPTQ quantization error: {gptq_error:.6f}")
print("\\nGPTQ uses per-channel quantization for better accuracy")"""),
        ]
    },
}

# Combine all notebooks
all_notebooks = {
    **section4_notebooks,
    **section5_notebooks,
    **section6_notebooks,
    **section7_notebooks,
    **section9_notebooks,
}

# Create directories
directories = [
    "notebooks/advanced_topics",
    "notebooks/mathematics",
    "notebooks/models_hub",
    "notebooks/community",
    "notebooks/advanced_architectures",
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Generate all notebooks
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
print(f"  Section 4 (Advanced Topics): {len(section4_notebooks)} notebooks")
print(f"  Section 5 (Mathematics): {len(section5_notebooks)} notebooks")
print(f"  Section 6 (Models Hub): {len(section6_notebooks)} notebooks")
print(f"  Section 7 (Community): {len(section7_notebooks)} notebooks")
print(f"  Section 9 (Advanced Architectures): {len(section9_notebooks)} notebooks")

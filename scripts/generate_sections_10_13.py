#!/usr/bin/env python3
"""Generate Sections 10-13 notebooks: Deployment, Cutting-Edge, Projects, About"""

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

# Section 10: Deployment & Production
section10_notebooks = {
    "deployment/01_serving_infrastructure.ipynb": {
        "title": "Module 10.1: Serving Infrastructure",
        "goal": "Set up production-ready serving infrastructure with vLLM and FastAPI",
        "time": "90 minutes",
        "concepts": [
            "vLLM production setup with PagedAttention",
            "Async API server with FastAPI",
            "Batch inference optimization",
            "PagedAttention KV cache management",
            "Throughput benchmarking"
        ],
        "cells": [
            create_cell("code", """# vLLM Production Setup
import subprocess
import json

# Install vLLM
# !pip install vllm fastapi uvicorn -q

print("vLLM provides:")
print("- PagedAttention for efficient KV cache")
print("- Continuous batching for high throughput")
print("- Async API support")
print("- Production-ready serving infrastructure")

# Example vLLM server setup
vllm_config = {
    "model": "HuggingFaceTB/SmolLM2-1.7B",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 2048,
    "dtype": "float16",
}

print(f"\\nvLLM Config: {json.dumps(vllm_config, indent=2)}")"""),
            create_cell("code", """# FastAPI Async Server Example
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI(title="SLM Inference API")

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

class GenerationResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    \"\"\"Generate text from prompt\"\"\"
    # In production, this would call vLLM engine
    # For demo purposes, we show the structure
    import time
    start = time.time()
    
    # Simulated generation
    await asyncio.sleep(0.1)  # Simulate inference time
    
    latency = (time.time() - start) * 1000
    
    return GenerationResponse(
        text="Generated text...",
        tokens_generated=request.max_tokens,
        latency_ms=latency
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

print("FastAPI server structure created!")
print("Run with: uvicorn main:app --host 0.0.0.0 --port 8000")"""),
            create_cell("code", """# Batch Inference Optimization
import torch
import time

def batch_inference(model, tokenizer, prompts, batch_size=8):
    \"\"\"Optimized batch inference\"\"\"
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
        
        # Decode
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    
    return results

print("Batch inference improves throughput by:")
print("- Processing multiple requests together")
print("- Better GPU utilization")
print("- Reduced overhead per request")"""),
        ]
    },
    "deployment/02_monitoring_observability.ipynb": {
        "title": "Module 10.2: Monitoring & Observability",
        "goal": "Set up comprehensive monitoring and observability for production deployments",
        "time": "90 minutes",
        "concepts": [
            "Prometheus metrics integration",
            "Grafana dashboard setup",
            "Structured logging with structlog",
            "Latency tracking (p50, p90, p99)",
            "Error rate monitoring",
            "GPU utilization tracking"
        ],
        "cells": [
            create_cell("code", """# Prometheus Metrics Integration
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
request_count = Counter('inference_requests_total', 'Total inference requests')
request_latency = Histogram('inference_latency_seconds', 'Inference latency')
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
error_count = Counter('inference_errors_total', 'Total inference errors')

def record_inference(latency_seconds, success=True):
    \"\"\"Record inference metrics\"\"\"
    request_count.inc()
    request_latency.observe(latency_seconds)
    
    if not success:
        error_count.inc()
    
    # Simulate GPU utilization
    gpu_utilization.set(85.5)

# Start Prometheus metrics server
# start_http_server(8000)

print("Prometheus metrics defined:")
print("- request_count: Total requests")
print("- request_latency: Latency histogram")
print("- gpu_utilization: GPU usage")
print("- error_count: Error tracking")"""),
            create_cell("code", """# Structured Logging with structlog
import structlog
import logging

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Example logging
logger.info("inference_request", 
            model="SmolLM-1.7B",
            prompt_length=50,
            max_tokens=100,
            user_id="user123")

logger.error("inference_failed",
             error="CUDA out of memory",
             model="SmolLM-1.7B",
             batch_size=8)

print("Structured logging provides:")
print("- JSON-formatted logs")
print("- Contextual information")
print("- Easy parsing and analysis")
print("- Better debugging")"""),
            create_cell("code", """# Latency Percentiles
import numpy as np
from collections import deque

class LatencyTracker:
    def __init__(self, window_size=1000):
        self.latencies = deque(maxlen=window_size)
    
    def record(self, latency_ms):
        self.latencies.append(latency_ms)
    
    def get_percentiles(self):
        if not self.latencies:
            return {}
        
        latencies_array = np.array(self.latencies)
        return {
            "p50": np.percentile(latencies_array, 50),
            "p90": np.percentile(latencies_array, 90),
            "p95": np.percentile(latencies_array, 95),
            "p99": np.percentile(latencies_array, 99),
            "mean": np.mean(latencies_array),
            "max": np.max(latencies_array),
        }

# Example usage
tracker = LatencyTracker()
for _ in range(100):
    # Simulate latency measurements
    latency = np.random.lognormal(mean=3, sigma=0.5)  # ms
    tracker.record(latency)

percentiles = tracker.get_percentiles()
print("Latency Percentiles (ms):")
for key, value in percentiles.items():
    print(f"  {key}: {value:.2f}")"""),
        ]
    },
    "deployment/03_cost_optimization.ipynb": {
        "title": "Module 10.3: Cost Optimization",
        "goal": "Optimize deployment costs through batching, quantization, and caching",
        "time": "75 minutes",
        "concepts": [
            "Cost breakdown analysis",
            "Batch inference strategies",
            "Dynamic batching implementation",
            "Model quantization for cost reduction",
            "Response caching",
            "Auto-scaling strategies"
        ],
        "cells": [
            create_cell("code", """# Cost Breakdown Analysis
def calculate_inference_cost(
    model_size_gb,
    requests_per_hour,
    avg_tokens_per_request,
    gpu_cost_per_hour=0.50,
    tokens_per_second=50
):
    \"\"\"Calculate inference costs\"\"\"
    # Compute time needed
    total_tokens = requests_per_hour * avg_tokens_per_request
    compute_hours = total_tokens / (tokens_per_second * 3600)
    
    # GPU costs
    gpu_cost = compute_hours * gpu_cost_per_hour
    
    # Memory costs (if using cloud storage)
    memory_cost_per_hour = model_size_gb * 0.01  # $0.01/GB/hour
    
    total_cost_per_hour = gpu_cost + memory_cost_per_hour
    total_cost_per_month = total_cost_per_hour * 24 * 30
    cost_per_request = total_cost_per_hour / requests_per_hour
    
    return {
        "gpu_cost_per_hour": gpu_cost,
        "memory_cost_per_hour": memory_cost_per_hour,
        "total_cost_per_hour": total_cost_per_hour,
        "total_cost_per_month": total_cost_per_month,
        "cost_per_request": cost_per_request,
    }

# Example calculation
costs = calculate_inference_cost(
    model_size_gb=4.5,  # SmolLM-1.7B in FP16
    requests_per_hour=1000,
    avg_tokens_per_request=200,
)

print("Cost Breakdown:")
for key, value in costs.items():
    if "cost" in key:
        print(f"  {key}: ${value:.4f}")"""),
            create_cell("code", """# Dynamic Batching
import asyncio
from collections import deque
from typing import List, Callable

class DynamicBatcher:
    def __init__(self, batch_size: int, max_wait_ms: int = 50):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
        self.processing = False
    
    async def add_request(self, request, callback: Callable):
        \"\"\"Add request to batch queue\"\"\"
        self.queue.append((request, callback))
        
        if len(self.queue) >= self.batch_size:
            await self._process_batch()
        elif not self.processing:
            asyncio.create_task(self._wait_and_process())
    
    async def _wait_and_process(self):
        \"\"\"Wait for max_wait_ms then process batch\"\"\"
        self.processing = True
        await asyncio.sleep(self.max_wait_ms / 1000)
        
        if self.queue:
            await self._process_batch()
        
        self.processing = False
    
    async def _process_batch(self):
        \"\"\"Process current batch\"\"\"
        if not self.queue:
            return
        
        batch = []
        callbacks = []
        
        while self.queue and len(batch) < self.batch_size:
            request, callback = self.queue.popleft()
            batch.append(request)
            callbacks.append(callback)
        
        # Process batch (simulated)
        results = await self._inference_batch(batch)
        
        # Call callbacks
        for callback, result in zip(callbacks, results):
            callback(result)
    
    async def _inference_batch(self, batch):
        \"\"\"Run inference on batch\"\"\"
        # Simulated batch inference
        await asyncio.sleep(0.1)
        return [f"result_{i}" for i in range(len(batch))]

print("Dynamic batching:")
print("- Collects requests over time window")
print("- Processes when batch is full or timeout")
print("- Improves throughput and reduces costs")"""),
            create_cell("code", """# Response Caching
from functools import lru_cache
import hashlib
import json

class ResponseCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _cache_key(self, prompt, **kwargs):
        \"\"\"Generate cache key from prompt and parameters\"\"\"
        key_data = {"prompt": prompt, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, prompt, **kwargs):
        \"\"\"Get cached response\"\"\"
        key = self._cache_key(prompt, **kwargs)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, prompt, response, **kwargs):
        \"\"\"Cache response\"\"\"
        key = self._cache_key(prompt, **kwargs)
        
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = response
    
    def hit_rate(self):
        \"\"\"Calculate cache hit rate\"\"\"
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

# Example usage
cache = ResponseCache()

# Cache a response
cache.set("What is AI?", "AI is artificial intelligence...", temperature=0.7)

# Retrieve from cache
cached = cache.get("What is AI?", temperature=0.7)
print(f"Cache hit: {cached is not None}")
print(f"Hit rate: {cache.hit_rate():.2%}")"""),
        ]
    },
}

# Section 11: Cutting-Edge
section11_notebooks = {
    "cutting_edge/01_bitnet_quantization.ipynb": {
        "title": "Module 11.1: BitNet - 1.58-bit Quantization",
        "goal": "Implement BitNet quantization with ternary weights (-1, 0, +1)",
        "time": "120 minutes",
        "concepts": [
            "Ternary weight training (-1, 0, +1)",
            "BitLinear layer implementation",
            "Straight-through estimator",
            "Training from scratch with BitNet",
            "Performance comparison (memory, speed, quality)",
            "Energy efficiency analysis"
        ],
        "cells": [
            create_cell("code", """# BitNet: 1.58-bit Quantization
import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Module):
    \"\"\"BitLinear layer with ternary weights (-1, 0, +1)\"\"\"
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Scale factor
        self.scale = nn.Parameter(torch.ones(1))
    
    def quantize_weights(self):
        \"\"\"Quantize weights to ternary (-1, 0, +1)\"\"\"
        # Calculate threshold (median of absolute values)
        abs_weights = self.weight.abs()
        threshold = abs_weights.median()
        
        # Quantize: sign(weight) if |weight| > threshold, else 0
        quantized = torch.sign(self.weight) * (abs_weights > threshold).float()
        
        # Scale to match original weight magnitude
        scale = self.weight.abs().mean() / quantized.abs().clamp(min=1e-8).mean()
        
        return quantized * scale
    
    def forward(self, x):
        \"\"\"Forward pass with straight-through estimator\"\"\"
        # During training: use full precision with STE
        # During inference: use quantized weights
        
        if self.training:
            # Straight-through estimator: quantize in forward, but use full precision gradient
            quantized = self.quantize_weights()
            return F.linear(x, quantized)
        else:
            # Inference: use quantized weights
            quantized = self.quantize_weights()
            return F.linear(x, quantized)

# Test BitLinear
bit_linear = BitLinear(128, 64)
x = torch.randn(2, 10, 128)

output = bit_linear(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"\\nBitNet benefits:")
print("- 1.58 bits per weight (ternary: -1, 0, +1)")
print("- ~16x memory reduction vs FP16")
print("- Faster inference (integer operations)")
print("- Energy efficient")"""),
            create_cell("code", """# Straight-Through Estimator (STE)
def ste_quantize(weights, bits=1):
    \"\"\"Quantize with straight-through estimator\"\"\"
    # Forward: quantize
    quantized = torch.sign(weights) * (weights.abs() > weights.abs().median()).float()
    
    # Backward: pass through gradient (no quantization in backward)
    return weights + (quantized - weights).detach()

# Example STE usage
x = torch.randn(4, 4, requires_grad=True)
x_quantized = ste_quantize(x)

print("Straight-Through Estimator:")
print("- Forward: quantized values")
print("- Backward: full precision gradients")
print("- Allows training with quantized weights")"""),
        ]
    },
    "cutting_edge/02_constitutional_ai_safety.ipynb": {
        "title": "Module 11.2: Constitutional AI & Safety",
        "goal": "Implement safety filters and constitutional AI workflow",
        "time": "90 minutes",
        "concepts": [
            "Constitutional AI workflow",
            "Critique and revision pipeline",
            "Safety filter implementation",
            "Toxicity detection",
            "PII detection",
            "Prompt injection prevention"
        ],
        "cells": [
            create_cell("code", """# Constitutional AI Workflow
class ConstitutionalAI:
    def __init__(self, model, constitution):
        self.model = model
        self.constitution = constitution  # List of principles
    
    def critique(self, response, prompt):
        \"\"\"Critique response against constitution\"\"\"
        critiques = []
        
        for principle in self.constitution:
            critique_prompt = f\"\"\"
Principle: {principle}
Response: {response}
Prompt: {prompt}

Does the response violate this principle? Explain.
\"\"\"
            # Check if response violates principle
            violation = self._check_violation(response, principle)
            if violation:
                critiques.append({
                    "principle": principle,
                    "violation": violation,
                    "severity": "high"
                })
        
        return critiques
    
    def revise(self, response, critiques):
        \"\"\"Revise response based on critiques\"\"\"
        if not critiques:
            return response
        
        revision_prompt = f\"\"\"
Original response: {response}
Critiques: {critiques}

Please revise the response to address these critiques while maintaining helpfulness.
\"\"\"
        revised = self.model.generate(revision_prompt)
        return revised
    
    def _check_violation(self, response, principle):
        \"\"\"Check if response violates principle\"\"\"
        # Simplified: check for keywords
        # In production, use a safety classifier
        return False

# Example constitution
constitution = [
    "Be helpful, harmless, and honest",
    "Do not generate harmful content",
    "Respect privacy and confidentiality",
    "Do not provide medical or legal advice",
]

print("Constitutional AI:")
print("- Critiques responses against principles")
print("- Revises responses to be safer")
print("- Iterative improvement process")"""),
            create_cell("code", """# Safety Filter Implementation
import re
from typing import List, Dict

class SafetyFilter:
    def __init__(self):
        # Toxicity keywords (simplified)
        self.toxicity_patterns = [
            r"\\b(kill|harm|violence)\\b",
            # Add more patterns
        ]
        
        # PII patterns
        self.pii_patterns = [
            r"\\b\\d{3}-\\d{2}-\\d{4}\\b",  # SSN
            r"\\b\\d{4}\\s?\\d{4}\\s?\\d{4}\\s?\\d{4}\\b",  # Credit card
            r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",  # Email
        ]
    
    def check_toxicity(self, text: str) -> Dict:
        \"\"\"Check for toxic content\"\"\"
        for pattern in self.toxicity_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return {
                    "is_toxic": True,
                    "reason": f"Matched pattern: {pattern}",
                    "severity": "high"
                }
        return {"is_toxic": False}
    
    def detect_pii(self, text: str) -> List[Dict]:
        \"\"\"Detect personally identifiable information\"\"\"
        pii_found = []
        
        for pattern in self.pii_patterns:
            matches = re.findall(pattern, text)
            if matches:
                pii_found.append({
                    "type": self._get_pii_type(pattern),
                    "matches": matches,
                    "count": len(matches)
                })
        
        return pii_found
    
    def _get_pii_type(self, pattern: str) -> str:
        \"\"\"Get PII type from pattern\"\"\"
        if "SSN" in pattern or "\\d{3}-\\d{2}-\\d{4}" in pattern:
            return "SSN"
        elif "credit" in pattern or "\\d{4}" in pattern:
            return "Credit Card"
        elif "@" in pattern:
            return "Email"
        return "Unknown"
    
    def filter_response(self, text: str) -> Dict:
        \"\"\"Comprehensive safety check\"\"\"
        toxicity = self.check_toxicity(text)
        pii = self.detect_pii(text)
        
        is_safe = not toxicity["is_toxic"] and len(pii) == 0
        
        return {
            "is_safe": is_safe,
            "toxicity": toxicity,
            "pii": pii,
            "filtered_text": self._redact_pii(text, pii) if pii else text
        }
    
    def _redact_pii(self, text: str, pii_list: List[Dict]) -> str:
        \"\"\"Redact PII from text\"\"\"
        filtered = text
        for pii_item in pii_list:
            for match in pii_item["matches"]:
                filtered = filtered.replace(match, "[REDACTED]")
        return filtered

# Example usage
filter = SafetyFilter()
test_text = "Contact me at john.doe@example.com or call 555-1234"

result = filter.filter_response(test_text)
print("Safety Filter Results:")
print(f"  Safe: {result['is_safe']}")
print(f"  PII detected: {len(result['pii'])}")
print(f"  Filtered text: {result['filtered_text']}")"""),
        ]
    },
    "cutting_edge/03_test_time_compute_scaling.ipynb": {
        "title": "Module 11.3: Test-Time Compute Scaling",
        "goal": "Implement test-time compute scaling techniques for better quality",
        "time": "90 minutes",
        "concepts": [
            "Chain-of-thought prompting",
            "Self-consistency (majority voting)",
            "Best-of-N sampling",
            "Interactive reasoning visualization",
            "Quality vs compute trade-offs"
        ],
        "cells": [
            create_cell("code", """# Chain-of-Thought Prompting
def chain_of_thought_prompt(question):
    \"\"\"Generate chain-of-thought prompt\"\"\"
    return f\"\"\"{question}

Let's think step by step:
1. First, I need to understand what is being asked.
2. Then, I'll break down the problem into smaller parts.
3. I'll solve each part systematically.
4. Finally, I'll combine the solutions.

Solution: \"\"\"

# Example
question = "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours?"
cot_prompt = chain_of_thought_prompt(question)

print("Chain-of-Thought Prompt:")
print(cot_prompt)"""),
            create_cell("code", """# Self-Consistency (Majority Voting)
import collections

def self_consistency_generate(model, tokenizer, prompt, num_samples=5):
    \"\"\"Generate multiple samples and use majority voting\"\"\"
    samples = []
    
    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        samples.append(response)
    
    # Extract answers (simplified - in practice, parse structured output)
    answers = [sample.split("Answer:")[-1].strip() for sample in samples]
    
    # Majority voting
    answer_counts = collections.Counter(answers)
    majority_answer = answer_counts.most_common(1)[0][0]
    confidence = answer_counts[majority_answer] / len(answers)
    
    return {
        "answer": majority_answer,
        "confidence": confidence,
        "all_samples": samples,
        "vote_distribution": dict(answer_counts)
    }

print("Self-Consistency:")
print("- Generate multiple samples")
print("- Use majority voting for final answer")
print("- Higher confidence with more agreement")"""),
            create_cell("code", """# Best-of-N Sampling
def best_of_n_sampling(model, tokenizer, prompt, n=10, scorer=None):
    \"\"\"Generate N samples and return the best one\"\"\"
    samples = []
    scores = []
    
    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        samples.append(response)
        
        # Score the response (e.g., using a reward model)
        if scorer:
            score = scorer(response)
        else:
            # Default: use length as proxy (longer = better for some tasks)
            score = len(response)
        
        scores.append(score)
    
    # Return best sample
    best_idx = scores.index(max(scores))
    
    return {
        "best_sample": samples[best_idx],
        "best_score": scores[best_idx],
        "all_samples": samples,
        "all_scores": scores,
    }

print("Best-of-N Sampling:")
print("- Generate N samples")
print("- Score each sample")
print("- Return the best one")
print("- Quality improves with N, but compute increases")"""),
        ]
    },
    "cutting_edge/04_long_context_techniques.ipynb": {
        "title": "Module 11.4: Long Context Techniques",
        "goal": "Extend context length beyond training with YaRN and streaming",
        "time": "90 minutes",
        "concepts": [
            "YaRN (RoPE extension) implementation",
            "Streaming inference with sliding window",
            "Sink token preservation",
            "Context length extension beyond training",
            "Memory-efficient long-context handling"
        ],
        "cells": [
            create_cell("code", """# YaRN: Yet another RoPE extensioN
import torch
import torch.nn as nn
import math

def apply_yarn_scaling(rope_freqs, scale_factor, original_max_len, new_max_len):
    \"\"\"Apply YaRN scaling to RoPE frequencies\"\"\"
    # YaRN scales frequencies to extend context
    # Scale factor typically: new_max_len / original_max_len
    
    # Low frequency components: scale down
    # High frequency components: keep original
    
    scaled_freqs = rope_freqs.clone()
    
    # Find transition point
    alpha = scale_factor
    beta = 32  # Hyperparameter
    
    # Scale low frequencies
    for i in range(len(rope_freqs)):
        if rope_freqs[i] < beta:
            scaled_freqs[i] = rope_freqs[i] / alpha
        else:
            # Keep high frequencies
            scaled_freqs[i] = rope_freqs[i]
    
    return scaled_freqs

# Example
original_max_len = 2048
new_max_len = 8192
scale_factor = new_max_len / original_max_len

# Simulated RoPE frequencies
rope_freqs = torch.linspace(0.1, 100, 64)

scaled_freqs = apply_yarn_scaling(rope_freqs, scale_factor, original_max_len, new_max_len)

print(f"YaRN Scaling:")
print(f"  Original max length: {original_max_len}")
print(f"  New max length: {new_max_len}")
print(f"  Scale factor: {scale_factor:.2f}")
print(f"  Low freq scaling: applied")
print(f"  High freq preservation: applied")"""),
            create_cell("code", """# Streaming Inference with Sliding Window
class SlidingWindowInference:
    def __init__(self, model, window_size=2048, stride=512):
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.kv_cache = None
    
    def process_long_sequence(self, tokens, max_new_tokens=100):
        \"\"\"Process long sequence with sliding window\"\"\"
        seq_len = len(tokens)
        
        if seq_len <= self.window_size:
            # Fits in one window
            return self.model.generate(tokens, max_new_tokens=max_new_tokens)
        
        # Sliding window approach
        outputs = []
        start_idx = 0
        
        while start_idx < seq_len:
            end_idx = min(start_idx + self.window_size, seq_len)
            window_tokens = tokens[start_idx:end_idx]
            
            # Process window
            window_output = self.model.generate(
                window_tokens,
                max_new_tokens=max_new_tokens if end_idx == seq_len else 0
            )
            
            outputs.append(window_output)
            
            # Move window
            start_idx += self.stride
        
        # Combine outputs
        return self._combine_outputs(outputs)
    
    def _combine_outputs(self, outputs):
        \"\"\"Combine outputs from multiple windows\"\"\"
        # Simple concatenation (in practice, use overlap handling)
        return torch.cat(outputs, dim=0)

print("Sliding Window Inference:")
print("- Process long sequences in chunks")
print("- Maintain context across windows")
print("- Memory efficient for long contexts")"""),
        ]
    },
    "cutting_edge/05_emergent_abilities_scaling.ipynb": {
        "title": "Module 11.5: Emergent Abilities & Scaling",
        "goal": "Understand and measure emergent abilities through scaling",
        "time": "75 minutes",
        "concepts": [
            "Measuring emergent abilities",
            "Chinchilla scaling law implementation",
            "Optimal compute allocation",
            "Model size vs training data trade-offs",
            "Emergence visualization across scales"
        ],
        "cells": [
            create_cell("code", """# Chinchilla Scaling Law
import numpy as np
import matplotlib.pyplot as plt

def chinchilla_optimal_allocation(compute_budget):
    \"\"\"Calculate optimal model size and training tokens using Chinchilla law\"\"\"
    # Chinchilla: N_opt = 0.6 * C^0.5, D_opt = 20 * C^0.5
    # Where C is compute in FLOPs, N is parameters, D is tokens
    
    # Convert compute to FLOPs (approximate)
    # 1 training step ≈ 6 * N * D FLOPs
    
    # Optimal allocation
    N_opt = 0.6 * (compute_budget ** 0.5)  # Parameters
    D_opt = 20 * (compute_budget ** 0.5)   # Tokens
    
    return {
        "parameters": int(N_opt),
        "tokens": int(D_opt),
        "compute_budget": compute_budget,
    }

# Example
compute_budgets = [1e18, 1e19, 1e20, 1e21]  # FLOPs

print("Chinchilla Optimal Allocation:")
for C in compute_budgets:
    allocation = chinchilla_optimal_allocation(C)
    print(f"\\nCompute: {C:.1e} FLOPs")
    print(f"  Optimal parameters: {allocation['parameters']:.2e}")
    print(f"  Optimal tokens: {allocation['tokens']:.2e}")"""),
            create_cell("code", """# Measuring Emergent Abilities
def measure_emergent_ability(model_sizes, benchmark_scores, threshold=0.5):
    \"\"\"Measure when abilities emerge (cross threshold)\"\"\"
    emergence_points = {}
    
    for benchmark_name, scores in benchmark_scores.items():
        # Find when score crosses threshold
        for i, (size, score) in enumerate(zip(model_sizes, scores)):
            if score >= threshold and (i == 0 or scores[i-1] < threshold):
                emergence_points[benchmark_name] = {
                    "model_size": size,
                    "score": score,
                    "index": i
                }
                break
    
    return emergence_points

# Example data
model_sizes = [135e6, 360e6, 1.7e9, 3.8e9, 7e9]  # Parameters
benchmark_scores = {
    "MMLU": [0.25, 0.32, 0.42, 0.55, 0.68],
    "GSM8K": [0.10, 0.18, 0.35, 0.52, 0.75],
    "HumanEval": [0.05, 0.12, 0.28, 0.45, 0.62],
}

emergence = measure_emergent_ability(model_sizes, benchmark_scores, threshold=0.5)

print("Emergent Abilities (crossing 50% threshold):")
for benchmark, info in emergence.items():
    print(f"  {benchmark}: {info['model_size']/1e9:.2f}B parameters (score: {info['score']:.2f})")"""),
        ]
    },
    "cutting_edge/06_multimodal_understanding.ipynb": {
        "title": "Module 11.6: Multimodal Understanding",
        "goal": "Work with vision-language models and cross-modal understanding",
        "time": "90 minutes",
        "concepts": [
            "Vision-language model architecture",
            "CLIP encoder integration",
            "Cross-attention implementation",
            "Image-text training pipeline",
            "Edge deployment for VLM (MiniCPM-V, MobileVLM)"
        ],
        "cells": [
            create_cell("code", """# Vision-Language Model Architecture
import torch
import torch.nn as nn

class VisionLanguageModel(nn.Module):
    \"\"\"Simple VLM architecture\"\"\"
    def __init__(self, vision_dim=768, text_dim=768, hidden_dim=1024):
        super().__init__()
        
        # Vision encoder (simulated - in practice use CLIP/ViT)
        self.vision_encoder = nn.Sequential(
            nn.Linear(224*224*3, vision_dim),  # Simplified
            nn.LayerNorm(vision_dim),
            nn.GELU(),
        )
        
        # Text encoder
        self.text_encoder = nn.Embedding(50000, text_dim)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Language model head
        self.lm_head = nn.Linear(hidden_dim, 50000)
    
    def forward(self, image, text_ids):
        \"\"\"Forward pass with image and text\"\"\"
        # Encode image
        image_features = self.vision_encoder(image)
        
        # Encode text
        text_features = self.text_encoder(text_ids)
        
        # Fuse features
        fused = self.fusion(torch.cat([image_features, text_features], dim=-1))
        
        # Cross-attention
        attn_out, _ = self.cross_attention(fused, fused, fused)
        
        # Language modeling
        logits = self.lm_head(attn_out)
        
        return logits

print("Vision-Language Model:")
print("- Vision encoder: processes images")
print("- Text encoder: processes text")
print("- Cross-attention: aligns modalities")
print("- LM head: generates text")"""),
            create_cell("code", """# CLIP-Style Contrastive Learning
def clip_loss(image_features, text_features, temperature=0.07):
    \"\"\"CLIP contrastive loss\"\"\"
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.t()) / temperature
    
    # Labels: diagonal (matched pairs)
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Cross-entropy loss (symmetric)
    loss_img = nn.functional.cross_entropy(logits, labels)
    loss_txt = nn.functional.cross_entropy(logits.t(), labels)
    
    return (loss_img + loss_txt) / 2

print("CLIP Contrastive Learning:")
print("- Learn aligned image-text embeddings")
print("- Maximize similarity for matched pairs")
print("- Minimize similarity for unmatched pairs")"""),
        ]
    },
}

# Section 12: Practical Projects
section12_notebooks = {
    "projects/01_code_assistant.ipynb": {
        "title": "Module 12.1: Build a Code Assistant",
        "goal": "Build a complete code completion assistant with fine-tuning",
        "time": "120 minutes",
        "concepts": [
            "Prepare coding dataset (The Stack)",
            "Fine-tune SmolLM with LoRA",
            "Code completion API",
            "VS Code extension integration",
            "Evaluation on HumanEval"
        ],
        "cells": [
            create_cell("code", """# Code Assistant Project Setup
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load The Stack dataset (code dataset)
# dataset = load_dataset("bigcode/the-stack", data_dir="data/python", split="train[:1%]")

print("Code Assistant Components:")
print("1. Dataset: The Stack (code dataset)")
print("2. Model: SmolLM-1.7B (base)")
print("3. Fine-tuning: LoRA on code")
print("4. API: FastAPI server")
print("5. Evaluation: HumanEval benchmark")

# Model setup
model_name = "HuggingFaceTB/SmolLM2-1.7B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"\\nBase model: {model_name}")"""),
            create_cell("code", """# Code Completion Function
def complete_code(model, tokenizer, prompt, max_tokens=100, temperature=0.2):
    \"\"\"Complete code from prompt\"\"\"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    completed = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completed

# Example
code_prompt = \"\"\"def fibonacci(n):
    \"\"\"Compute the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return\"\"\"

print("Code Completion Example:")
print(f"Prompt: {code_prompt}")
print("\\n(Would generate completion here)")"""),
            create_cell("code", """# HumanEval Evaluation
def evaluate_humaneval(model, tokenizer, problems):
    \"\"\"Evaluate on HumanEval benchmark\"\"\"
    results = []
    
    for problem in problems:
        prompt = problem["prompt"]
        test_cases = problem["test"]
        
        # Generate solution
        solution = complete_code(model, tokenizer, prompt)
        
        # Extract function code
        # (In practice, parse and execute test cases)
        
        results.append({
            "problem_id": problem["task_id"],
            "solution": solution,
            "passed": False,  # Would check test cases
        })
    
    # Calculate pass rate
    pass_rate = sum(r["passed"] for r in results) / len(results)
    
    return {
        "pass_rate": pass_rate,
        "results": results
    }

print("HumanEval Evaluation:")
print("- 164 programming problems")
print("- Tests code generation quality")
print("- Pass rate: percentage of problems solved")"""),
        ]
    },
    "projects/02_personal_knowledge_base.ipynb": {
        "title": "Module 12.2: Personal Knowledge Base (RAG)",
        "goal": "Build a complete RAG system for personal knowledge management",
        "time": "120 minutes",
        "concepts": [
            "Document loading and chunking",
            "Embedding generation",
            "FAISS vector store setup",
            "Multi-hop RAG implementation",
            "Query interface",
            "Performance optimization"
        ],
        "cells": [
            create_cell("code", """# Personal Knowledge Base RAG System
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np

class PersonalKnowledgeBase:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.documents = []
    
    def add_documents(self, texts):
        \"\"\"Add documents to knowledge base\"\"\"
        # Chunk documents
        chunks = []
        for text in texts:
            chunks.extend(self.text_splitter.split_text(text))
        
        # Generate embeddings
        chunk_embeddings = self.embeddings.embed_documents(chunks)
        
        # Create FAISS index
        dimension = len(chunk_embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings
        embeddings_array = np.array(chunk_embeddings).astype('float32')
        index.add(embeddings_array)
        
        self.vector_store = index
        self.documents = chunks
        
        print(f"Added {len(chunks)} chunks to knowledge base")
    
    def search(self, query, top_k=5):
        \"\"\"Search knowledge base\"\"\"
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.vector_store.search(query_vector, top_k)
        
        # Retrieve documents
        results = [self.documents[idx] for idx in indices[0]]
        
        return results

# Example usage
kb = PersonalKnowledgeBase()
documents = [
    "Python is a programming language...",
    "Machine learning uses algorithms...",
    # Add more documents
]

# kb.add_documents(documents)
# results = kb.search("What is Python?")

print("Personal Knowledge Base:")
print("- Document chunking")
print("- Embedding generation")
print("- FAISS vector search")
print("- Query interface")"""),
            create_cell("code", """# Multi-Hop RAG
class MultiHopRAG:
    def __init__(self, knowledge_base, llm):
        self.kb = knowledge_base
        self.llm = llm
    
    def query(self, question, max_hops=3):
        \"\"\"Multi-hop reasoning with RAG\"\"\"
        context = []
        current_query = question
        
        for hop in range(max_hops):
            # Retrieve relevant documents
            docs = self.kb.search(current_query, top_k=3)
            context.extend(docs)
            
            # Generate refined query or answer
            if hop < max_hops - 1:
                # Refine query for next hop
                refinement_prompt = f\"\"\"
Question: {question}
Context so far: {''.join(context[-3:])}
What additional information do we need to answer this question?
\"\"\"
                current_query = self.llm.generate(refinement_prompt)
            else:
                # Final answer
                answer_prompt = f\"\"\"
Question: {question}
Context: {''.join(context)}
Answer:
\"\"\"
                answer = self.llm.generate(answer_prompt)
                return answer
        
        return "Unable to answer"

print("Multi-Hop RAG:")
print("- Iterative retrieval")
print("- Query refinement")
print("- Better for complex questions")"""),
        ]
    },
    "projects/03_function_calling_agent.ipynb": {
        "title": "Module 12.3: Function-Calling Agent",
        "goal": "Build a complete function-calling agent with tool orchestration",
        "time": "120 minutes",
        "concepts": [
            "Tool definition schema",
            "ReAct pattern implementation",
            "Multi-tool orchestration",
            "Error handling and retries",
            "Agent evaluation framework"
        ],
        "cells": [
            create_cell("code", """# Function-Calling Agent
from typing import List, Dict, Callable, Any
import json

class Tool:
    def __init__(self, name: str, description: str, func: Callable, schema: Dict):
        self.name = name
        self.description = description
        self.func = func
        self.schema = schema
    
    def call(self, **kwargs):
        return self.func(**kwargs)
    
    def to_json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.schema
        }

class FunctionCallingAgent:
    def __init__(self, model, tools: List[Tool]):
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.conversation_history = []
    
    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def get_tools_description(self):
        \"\"\"Get tools description for prompt\"\"\"
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\\n".join(descriptions)
    
    def execute(self, user_query: str, max_iterations=5):
        \"\"\"Execute agent with ReAct pattern\"\"\"
        self.conversation_history.append({"role": "user", "content": user_query})
        
        for iteration in range(max_iterations):
            # Generate response (with tool calling)
            tools_desc = self.get_tools_description()
            prompt = f\"\"\"
Available tools:
{tools_desc}

User query: {user_query}
Conversation history: {self.conversation_history}

Generate response. If you need to use a tool, format as:
TOOL_CALL: tool_name(argument1=value1, argument2=value2)
\"\"\"
            
            response = self.model.generate(prompt)
            
            # Parse tool calls
            if "TOOL_CALL:" in response:
                tool_call = self._parse_tool_call(response)
                if tool_call:
                    tool_name, args = tool_call
                    if tool_name in self.tools:
                        try:
                            result = self.tools[tool_name].call(**args)
                            self.conversation_history.append({
                                "role": "assistant",
                                "content": f"Tool {tool_name} returned: {result}"
                            })
                            continue
                        except Exception as e:
                            self.conversation_history.append({
                                "role": "error",
                                "content": f"Tool {tool_name} failed: {str(e)}"
                            })
            
            # Final answer
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        return "Max iterations reached"
    
    def _parse_tool_call(self, text: str):
        \"\"\"Parse tool call from text\"\"\"
        # Simplified parser
        if "TOOL_CALL:" in text:
            parts = text.split("TOOL_CALL:")[1].strip().split("(")
            tool_name = parts[0].strip()
            args_str = "(".join(parts[1:]).rstrip(")")
            # Parse arguments (simplified)
            args = {}
            return tool_name, args
        return None

# Example tools
def calculator(operation: str, a: float, b: float) -> float:
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    return 0.0

calc_tool = Tool(
    name="calculator",
    description="Perform basic math operations",
    func=calculator,
    schema={
        "type": "object",
        "properties": {
            "operation": {"type": "string"},
            "a": {"type": "number"},
            "b": {"type": "number"}
        }
    }
)

print("Function-Calling Agent:")
print("- Tool definition and registration")
print("- ReAct pattern (Reasoning + Acting)")
print("- Multi-tool orchestration")
print("- Error handling")"""),
        ]
    },
}

# Section 13: About & Resources
section13_notebooks = {
    "about/01_resources_reference.ipynb": {
        "title": "Module 13.1: Resources & Quick Reference",
        "goal": "Comprehensive resource compilation and quick reference guides",
        "time": "30 minutes",
        "concepts": [
            "Official documentation links",
            "Must-read papers list",
            "Community resources",
            "Model selection flowchart",
            "Optimization techniques summary",
            "Quick reference tables"
        ],
        "cells": [
            create_cell("code", """# Resources & Quick Reference

# Official Documentation
documentation = {
    "Transformers": "https://huggingface.co/docs/transformers",
    "vLLM": "https://docs.vllm.ai",
    "PEFT": "https://huggingface.co/docs/peft",
    "TRL": "https://huggingface.co/docs/trl",
    "PyTorch": "https://pytorch.org/docs",
}

# Must-Read Papers
papers = {
    "Attention Is All You Need": "https://arxiv.org/abs/1706.03762",
    "LoRA: Low-Rank Adaptation": "https://arxiv.org/abs/2106.09685",
    "Chinchilla Scaling Laws": "https://arxiv.org/abs/2203.15556",
    "Mamba: Linear-Time Sequence Modeling": "https://arxiv.org/abs/2312.00752",
    "BitNet: Scaling 1-bit Transformers": "https://arxiv.org/abs/2310.11453",
    "Constitutional AI": "https://arxiv.org/abs/2212.08073",
}

# Community Resources
community = {
    "Discord": "https://discord.gg/slmhub",
    "GitHub": "https://github.com/lmfast/slmhub",
    "HuggingFace": "https://huggingface.co",
    "Papers with Code": "https://paperswithcode.com",
}

print("Resources:")
print("\\nDocumentation:")
for name, url in documentation.items():
    print(f"  {name}: {url}")

print("\\nMust-Read Papers:")
for name, url in papers.items():
    print(f"  {name}: {url}")

print("\\nCommunity:")
for name, url in community.items():
    print(f"  {name}: {url}")"""),
            create_cell("code", """# Model Selection Flowchart
def select_model(use_case, memory_gb, latency_ms=None, accuracy_requirement="medium"):
    \"\"\"Model selection helper\"\"\"
    recommendations = []
    
    if memory_gb < 1:
        recommendations.append("SmolLM-135M (INT4)")
    elif memory_gb < 2:
        recommendations.append("SmolLM-360M (INT4)")
    elif memory_gb < 4:
        recommendations.append("SmolLM-1.7B (INT4)")
    elif memory_gb < 8:
        recommendations.append("SmolLM-1.7B (FP16) or Phi-3-mini (INT4)")
    else:
        recommendations.append("Phi-3-mini (FP16) or larger models")
    
    if use_case == "code":
        recommendations.append("Consider: StarCoder2, CodeQwen")
    elif use_case == "math":
        recommendations.append("Consider: DeepSeek-Math")
    elif use_case == "multilingual":
        recommendations.append("Consider: Qwen2.5, XGLM")
    
    return recommendations

print("Model Selection Guide:")
print(select_model("general", memory_gb=4, accuracy_requirement="high"))"""),
            create_cell("code", """# Quick Reference Tables

# Quantization Comparison
quantization_table = {
    "FP32": {"bits": 32, "memory_reduction": "1x", "quality": "100%", "speed": "1x"},
    "FP16": {"bits": 16, "memory_reduction": "2x", "quality": "99%", "speed": "1.5x"},
    "INT8": {"bits": 8, "memory_reduction": "4x", "quality": "95%", "speed": "2x"},
    "INT4": {"bits": 4, "memory_reduction": "8x", "quality": "90%", "speed": "3x"},
    "BitNet": {"bits": 1.58, "memory_reduction": "16x", "quality": "85%", "speed": "4x"},
}

print("Quantization Comparison:")
for method, specs in quantization_table.items():
    print(f"  {method}: {specs['bits']} bits, {specs['memory_reduction']} memory, {specs['quality']} quality")

# Optimization Techniques
optimization_techniques = {
    "LoRA": "Parameter-efficient fine-tuning",
    "QLoRA": "LoRA with 4-bit quantization",
    "Gradient Checkpointing": "Trade compute for memory",
    "Mixed Precision": "FP16/BF16 training",
    "FlashAttention": "Memory-efficient attention",
    "KV Cache": "Faster generation",
    "Speculative Decoding": "2-3x speedup",
}

print("\\nOptimization Techniques:")
for technique, description in optimization_techniques.items():
    print(f"  {technique}: {description}")"""),
        ]
    },
}

# Combine all notebooks
all_notebooks = {
    **section10_notebooks,
    **section11_notebooks,
    **section12_notebooks,
    **section13_notebooks,
}

# Create directories
directories = [
    "notebooks/deployment",
    "notebooks/cutting_edge",
    "notebooks/projects",
    "notebooks/about",
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
print(f"  Section 10 (Deployment): {len(section10_notebooks)} notebooks")
print(f"  Section 11 (Cutting-Edge): {len(section11_notebooks)} notebooks")
print(f"  Section 12 (Projects): {len(section12_notebooks)} notebooks")
print(f"  Section 13 (About): {len(section13_notebooks)} notebooks")

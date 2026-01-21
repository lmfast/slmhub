#!/usr/bin/env python3
"""Model database utilities for loading and querying model metadata"""

import yaml
import json
from typing import Dict, List, Optional
from pathlib import Path

def load_model_database(db_path: str = "models/generated") -> List[Dict]:
    """Load all model YAML files from database directory"""
    models = []
    db_dir = Path(db_path)
    
    if not db_dir.exists():
        return models
    
    for yaml_file in db_dir.glob("*.md"):
        try:
            with open(yaml_file, 'r') as f:
                content = f.read()
                # Extract YAML frontmatter if present
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        yaml_content = parts[1]
                        model_data = yaml.safe_load(yaml_content)
                        if model_data:
                            models.append(model_data)
        except Exception as e:
            print(f"Error loading {yaml_file}: {e}")
    
    return models

def filter_models(
    models: List[Dict],
    max_params: Optional[int] = None,
    min_params: Optional[int] = None,
    license_type: Optional[str] = None,
    min_mmlu: Optional[float] = None,
    tags: Optional[List[str]] = None,
) -> List[Dict]:
    """Filter models by various criteria"""
    filtered = models.copy()
    
    if max_params:
        filtered = [m for m in filtered if m.get('parameters', 0) <= max_params]
    
    if min_params:
        filtered = [m for m in filtered if m.get('parameters', 0) >= min_params]
    
    if license_type:
        filtered = [m for m in filtered if license_type.lower() in m.get('license', '').lower()]
    
    if min_mmlu:
        benchmarks = m.get('benchmarks', {})
        mmlu_score = benchmarks.get('mmlu', 0)
        filtered = [m for m in filtered if mmlu_score >= min_mmlu]
    
    if tags:
        filtered = [m for m in filtered if any(tag in m.get('tags', []) for tag in tags)]
    
    return filtered

def calculate_memory_requirements(
    params: int,
    precision: str = "fp16",
    batch_size: int = 1,
    seq_len: int = 2048,
) -> Dict[str, float]:
    """Calculate memory requirements for a model"""
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }.get(precision, 2)
    
    # Model weights
    model_memory_gb = params * bytes_per_param / 1e9
    
    # KV cache (approximate)
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

def validate_model_entry(model_data: Dict) -> tuple[bool, str]:
    """Validate a model entry against schema"""
    required_fields = ["name", "parameters", "license"]
    missing = [field for field in required_fields if field not in model_data]
    
    if missing:
        return False, f"Missing required fields: {missing}"
    
    # Validate parameter count
    if not isinstance(model_data["parameters"], int) or model_data["parameters"] <= 0:
        return False, "Parameters must be a positive integer"
    
    return True, "Valid model entry"

def generate_comparison_table(models: List[Dict], metrics: List[str] = None) -> str:
    """Generate a markdown comparison table"""
    if not models:
        return "No models to compare"
    
    if metrics is None:
        metrics = ["name", "parameters", "license"]
    
    # Build table header
    header = "| " + " | ".join(metrics) + " |"
    separator = "| " + " | ".join(["---"] * len(metrics)) + " |"
    
    # Build table rows
    rows = []
    for model in models:
        row = []
        for metric in metrics:
            value = model.get(metric, "N/A")
            if isinstance(value, dict):
                value = json.dumps(value)
            row.append(str(value))
        rows.append("| " + " | ".join(row) + " |")
    
    return "\n".join([header, separator] + rows)

if __name__ == "__main__":
    # Example usage
    models = load_model_database()
    print(f"Loaded {len(models)} models")
    
    # Filter example
    small_models = filter_models(models, max_params=2_000_000_000)
    print(f"Found {len(small_models)} models with <2B parameters")

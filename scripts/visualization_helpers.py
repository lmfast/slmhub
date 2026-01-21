#!/usr/bin/env python3
"""Common visualization helpers for notebooks"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional

def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    title: str = "Attention Heatmap",
    figsize: tuple = (10, 8),
):
    """Plot attention weight heatmap"""
    plt.figure(figsize=figsize)
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.5
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Key (attending to)', fontsize=12)
    plt.ylabel('Query (attending from)', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_learning_curve(
    losses: List[float],
    title: str = "Learning Curve",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    figsize: tuple = (10, 5),
):
    """Plot training loss curve"""
    plt.figure(figsize=figsize)
    plt.plot(losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_radar_chart(
    data: Dict[str, List[float]],
    categories: List[str],
    title: str = "Radar Chart",
    figsize: tuple = (8, 8),
):
    """Plot radar chart for model comparison"""
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete circle
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    for label, values in data.items():
        values = list(values) + [values[0]]  # Complete circle
        ax.plot(angles, values, 'o-', linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(title)
    plt.show()

def plot_model_comparison(
    models: List[str],
    metrics: Dict[str, List[float]],
    title: str = "Model Comparison",
    figsize: tuple = (12, 6),
):
    """Plot bar chart comparing models across metrics"""
    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - len(metrics) / 2) * width + width / 2
        ax.bar(x + offset, values, width, label=metric_name)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_distribution(
    data: np.ndarray,
    title: str = "Distribution",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    bins: int = 50,
    figsize: tuple = (10, 6),
):
    """Plot histogram of data distribution"""
    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.show()

def plot_time_series(
    x: List[float],
    y: List[float],
    title: str = "Time Series",
    xlabel: str = "Time",
    ylabel: str = "Value",
    figsize: tuple = (10, 6),
):
    """Plot time series data"""
    plt.figure(figsize=figsize)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_multi_head_attention(
    attention_weights: np.ndarray,
    tokens: List[str],
    num_heads: int,
    title: str = "Multi-Head Attention",
    figsize: tuple = (18, 9),
):
    """Plot attention weights for all heads"""
    fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    
    for head in range(num_heads):
        attn_matrix = attention_weights[0, head] if len(attention_weights.shape) == 4 else attention_weights[head]
        sns.heatmap(
            attn_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            ax=axes[head],
            cbar=True,
            vmin=0,
            vmax=1,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Weight'}
        )
        axes[head].set_title(f'Head {head + 1}', fontsize=11, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

# Configure default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

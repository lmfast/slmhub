# SLM Hub - Developer-Centric Platform
## Complete Technical Specification & Implementation Guide

**Version**: 4.0 (GitBook Optimized)  
**Last Updated**: January 19, 2026  
**Focus**: Developer-First | Minimal Design | Automated | Always Current

---

## 🎯 Executive Summary

### Vision
Build the **developer's first stop** for Small Language Models - practical, minimal, always up-to-date.

### Core Philosophy (2026)

```
DEVELOPER-FIRST
├── No marketing fluff
├── Code that runs → Copy, paste, deploy
├── Real benchmarks → Not inflated claims
└── Honest limitations → What works, what doesn't

MINIMAL DESIGN
├── Content over chrome
├── Fast load times (<2s)
├── Clean typography
└── Zero distractions

ALWAYS CURRENT
├── Automated model updates (daily)
├── Code validation (on every PR)
├── Dead link checking (weekly)
└── Community-driven corrections

LEARN BY DOING
├── Every concept → Colab notebook
├── Interactive examples → Try before reading
├── Progressive complexity → Skip what you know
└── Real production code → Not toy examples
```

### What We're NOT Building

❌ Model marketplace (Hugging Face exists)  
❌ Benchmarking service (too many already)  
❌ Hosting platform (use Ollama/vLLM)  
❌ Generic ML tutorials (Fast.ai, Made with ML)

### What We ARE Building

✅ **Practical SLM guide** - "How do I deploy Phi-4 on a Raspberry Pi?"  
✅ **Living documentation** - Updates automatically from HF Hub  
✅ **Code-first learning** - Working examples, not theory  
✅ **Community-driven** - Open source, PRs welcome

---

## 📊 Technology Stack

### Documentation Platform: GitBook

**Why GitBook?**

| Feature | GitBook | Alternatives | Winner |
|---------|---------|--------------|--------|
| **GitHub Sync** | Native bidirectional | External tools | ✅ GitBook |
| **Zero Build** | Push MD → Live site | Requires build step | ✅ GitBook |
| **AI Search** | Built-in (2026) | Need to integrate | ✅ GitBook |
| **Free Tier** | Unlimited public docs | Limited pages | ✅ GitBook |
| **Markdown** | Full support + extensions | Varies | ✅ GitBook |
| **Version Control** | Git-native | Database-backed | ✅ GitBook |

**GitBook Features We'll Use:**

1. **Git Sync** - Bidirectional sync with GitHub
2. **GitBook AI** - Built-in search & Q&A (no setup)
3. **Spaces** - Logical content organization
4. **Change Requests** - PR-like review workflow
5. **Insights** - Track what developers read
6. **Custom Domain** - `slmhub.dev`
7. **API Access** - Programmatic content updates

### Core Technology Decisions

```yaml
# Platform
documentation: GitBook (Markdown + Git)
source_control: GitHub (public repo)
automation: GitHub Actions (free for public repos)
tutorials: Google Colab (free GPU, zero setup)
model_data: Hugging Face API (official source)
analytics: Plausible (privacy-friendly, GDPR compliant)
search: GitBook AI (built-in, no config)
comments: GitHub Discussions (native integration)
newsletter: Buttondown (simple, affordable)

# Optional Enhancements
database: Supabase (PostgreSQL, free tier)
cdn: Cloudflare (caching, DDoS protection)
forms: Tally (simple, free tier)
monitoring: Better Uptime (status page)
```

---

## 🏗️ Information Architecture

### Site Structure (GitBook Spaces)

```
slm-hub/
│
├── 🏠 Home (README.md)
│   ├── What are SLMs?
│   ├── Why SLMs Matter in 2026
│   ├── Quick Start (< 5 min)
│   └── Community Showcase
│
├── 📚 Learn
│   ├── Fundamentals/
│   │   ├── slm-vs-llm.md
│   │   ├── tokenization.md
│   │   ├── quantization.md
│   │   ├── fine-tuning-basics.md
│   │   └── prompt-engineering.md
│   │
│   ├── Tutorials/
│   │   ├── Beginner/
│   │   │   ├── hello-slm.md (WebGPU demo)
│   │   │   ├── run-phi4-locally.md
│   │   │   └── first-fine-tune.md
│   │   │
│   │   ├── Intermediate/
│   │   │   ├── rag-with-slm.md
│   │   │   ├── deploy-raspberry-pi.md
│   │   │   └── ollama-vs-vllm.md
│   │   │
│   │   └── Advanced/
│   │       ├── distributed-training.md
│   │       ├── quantization-deep-dive.md
│   │       └── production-deployment.md
│   │
│   └── Colab Notebooks/
│       ├── all-notebooks.md (index)
│       └── [auto-generated from /notebooks]
│
├── 🤖 Models
│   ├── directory.md (auto-updated daily)
│   ├── comparison-tool.md (interactive)
│   │
│   ├── Featured/
│   │   ├── phi-4.md
│   │   ├── qwen3-8b.md
│   │   ├── gemma-3-9b.md
│   │   └── smollm3-1.7b.md
│   │
│   ├── By Task/
│   │   ├── text-generation.md
│   │   ├── code-generation.md
│   │   ├── multimodal.md
│   │   └── reasoning.md
│   │
│   └── By Hardware/
│       ├── raspberry-pi.md
│       ├── jetson.md
│       ├── mobile.md
│       └── gpu.md
│
├── 🚀 Deploy
│   ├── Quick Start/
│   │   ├── ollama.md
│   │   ├── vllm.md
│   │   ├── llama-cpp.md
│   │   └── transformers-js.md
│   │
│   ├── Deployment Patterns/
│   │   ├── edge-first-cloud-fallback.md
│   │   ├── hybrid-slm-llm-routing.md
│   │   ├── rag-enhanced-slm.md
│   │   └── multi-expert-system.md
│   │
│   ├── Hardware Guides/
│   │   ├── raspberry-pi-5.md
│   │   ├── jetson-nano.md
│   │   ├── ios-deployment.md
│   │   └── android-deployment.md
│   │
│   └── Production/
│       ├── monitoring.md
│       ├── scaling.md
│       ├── security.md
│       └── cost-optimization.md
│
├── 🛠️ Tools
│   ├── cost-calculator.md
│   ├── model-selector.md
│   ├── hardware-picker.md
│   ├── tokenizer-playground.md
│   └── benchmark-runner.md
│
├── 💡 Use Cases
│   ├── by-industry.md
│   ├── case-studies.md
│   └── community-projects.md
│
└── 🤝 Community
    ├── contributing.md
    ├── discussions.md (→ GitHub Discussions)
    ├── showcase.md (user submissions)
    ├── newsletter.md (→ Buttondown)
    └── faq.md
```

### Navigation Principles

**Top-Level Navigation (GitBook sidebar):**
```
Home → Learn → Models → Deploy → Tools → Community
```

**Every Page Has:**
- Breadcrumbs (`Home > Deploy > Hardware > Raspberry Pi 5`)
- Next/Previous navigation
- "Edit on GitHub" link
- Table of contents (auto-generated)
- Last updated timestamp

**Search Strategy:**
- GitBook AI search (primary)
- Algolia (if needed for advanced filtering)
- Category-specific search scopes

---

## 🤖 Automation Strategy

### Goal: Zero Manual Work for Updates

**What Gets Automated:**

```yaml
Daily (2 AM UTC):
  - Fetch latest models from Hugging Face
  - Update model cards with new benchmarks
  - Refresh download counts & popularity
  - Check for new model releases
  - Update "What's New" section

On Every PR:
  - Validate all code examples
  - Test Colab notebooks
  - Check Markdown formatting
  - Validate links (internal)
  - Spell check

Weekly (Monday 8 AM UTC):
  - Check all external links
  - Validate image URLs
  - Update dependencies in code examples
  - Generate sitemap
  - Update community showcase

Monthly:
  - Archive old discussions
  - Update model comparison charts
  - Refresh benchmarks
  - Generate "State of SLMs" report
```

### GitHub Actions Workflows

#### 1. Model Data Sync (Daily)

**File**: `.github/workflows/sync-models.yml`

```yaml
name: Sync Model Data

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:  # Manual trigger
  
permissions:
  contents: write

jobs:
  sync-models:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install huggingface_hub requests pyyaml jinja2
      
      - name: Fetch model data
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python scripts/fetch_models.py
      
      - name: Generate model pages
        run: |
          python scripts/generate_model_pages.py
      
      - name: Commit changes
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "🤖 Auto-update: Model data"
          file_pattern: 'models/*.md data/*.json'
          commit_author: SLM Bot <bot@slmhub.dev>
```

**Script**: `scripts/fetch_models.py`

```python
"""Fetch latest SLM data from Hugging Face"""
import os
import json
import yaml
from datetime import datetime
from huggingface_hub import HfApi
from typing import List, Dict, Optional

# Featured SLMs to track
FEATURED_MODELS = [
    "microsoft/Phi-4",
    "Qwen/Qwen3-8B",
    "google/gemma-3-9b",
    "HuggingFaceTB/SmolLM3-1.7B",
    "mistralai/Ministral-3-3B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "bigcode/starcoder2-7b",
    "google/gemma-3n-e2b-it",
]

# Task-based filters (search for new models)
TASK_FILTERS = {
    "text-generation": {"limit": 50, "size_max": 15_000_000_000},
    "text-to-code": {"limit": 30, "size_max": 15_000_000_000},
    "image-text-to-text": {"limit": 20, "size_max": 10_000_000_000},
}

class ModelFetcher:
    def __init__(self, hf_token: Optional[str] = None):
        self.api = HfApi(token=hf_token)
        
    def fetch_model_info(self, model_id: str) -> Optional[Dict]:
        """Fetch comprehensive model information"""
        try:
            model_info = self.api.model_info(model_id, files_metadata=True)
            
            # Extract key information
            data = {
                'id': model_id,
                'name': model_id.split('/')[-1],
                'author': model_info.author,
                'downloads': model_info.downloads,
                'likes': model_info.likes,
                'tags': model_info.tags or [],
                'pipeline_tag': model_info.pipeline_tag,
                'library_name': model_info.library_name,
                'created_at': model_info.created_at.isoformat() if model_info.created_at else None,
                'last_modified': model_info.last_modified.isoformat() if model_info.last_modified else None,
                'updated_at': datetime.now().isoformat(),
            }
            
            # Extract card data (benchmarks, params, etc.)
            if hasattr(model_info, 'card_data') and model_info.card_data:
                card_data = model_info.card_data
                data['card_data'] = {
                    'language': card_data.get('language', []),
                    'license': card_data.get('license', 'unknown'),
                    'model-index': card_data.get('model-index', []),
                }
                
                # Extract benchmarks
                benchmarks = {}
                for model_index in card_data.get('model-index', []):
                    for result in model_index.get('results', []):
                        for metric in result.get('metrics', []):
                            bench_name = result.get('dataset', {}).get('name', '')
                            metric_name = metric.get('name', '')
                            if bench_name and metric_name:
                                key = f"{bench_name}_{metric_name}"
                                benchmarks[key] = metric.get('value')
                
                data['benchmarks'] = benchmarks
            
            # Calculate model size (approximate from safetensors)
            total_size = 0
            if hasattr(model_info, 'safetensors') and model_info.safetensors:
                total_size = sum(
                    f.size for files in model_info.safetensors.values() 
                    for f in files
                )
            data['size_bytes'] = total_size
            data['size_gb'] = round(total_size / (1024**3), 2)
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching {model_id}: {e}")
            return None
    
    def discover_new_models(self, task: str, config: Dict) -> List[str]:
        """Discover new models by task"""
        try:
            models = self.api.list_models(
                filter=task,
                sort="lastModified",
                direction=-1,
                limit=config['limit']
            )
            
            # Filter by size
            filtered = []
            for model in models:
                if hasattr(model, 'safetensors') and model.safetensors:
                    total_size = sum(
                        f.size for files in model.safetensors.values() 
                        for f in files
                    )
                    if total_size <= config['size_max']:
                        filtered.append(model.modelId)
            
            return filtered
            
        except Exception as e:
            print(f"❌ Error discovering {task} models: {e}")
            return []

def main():
    """Main execution"""
    # Initialize fetcher
    hf_token = os.getenv('HF_TOKEN')
    fetcher = ModelFetcher(hf_token)
    
    # Fetch featured models
    print("📦 Fetching featured models...")
    featured_data = []
    for model_id in FEATURED_MODELS:
        print(f"  → {model_id}")
        data = fetcher.fetch_model_info(model_id)
        if data:
            data['featured'] = True
            featured_data.append(data)
    
    print(f"✅ Fetched {len(featured_data)} featured models")
    
    # Discover new models
    print("\n🔍 Discovering new models...")
    discovered_ids = set()
    for task, config in TASK_FILTERS.items():
        print(f"  → {task}")
        new_ids = fetcher.discover_new_models(task, config)
        discovered_ids.update(new_ids)
    
    # Remove already-tracked models
    discovered_ids -= set(FEATURED_MODELS)
    
    # Fetch discovered models
    discovered_data = []
    for model_id in list(discovered_ids)[:20]:  # Limit to 20 new
        data = fetcher.fetch_model_info(model_id)
        if data:
            data['featured'] = False
            discovered_data.append(data)
    
    print(f"✅ Discovered {len(discovered_data)} new models")
    
    # Combine all data
    all_models = featured_data + discovered_data
    
    # Save to files
    os.makedirs('data', exist_ok=True)
    
    # JSON (for programmatic access)
    with open('data/models.json', 'w') as f:
        json.dump(all_models, f, indent=2)
    
    # YAML (for GitBook)
    with open('data/models.yaml', 'w') as f:
        yaml.dump(all_models, f, default_flow_style=False)
    
    # Summary stats
    with open('data/stats.json', 'w') as f:
        stats = {
            'total_models': len(all_models),
            'featured': len(featured_data),
            'discovered': len(discovered_data),
            'last_updated': datetime.now().isoformat(),
            'total_downloads': sum(m['downloads'] for m in all_models),
            'total_likes': sum(m['likes'] for m in all_models),
        }
        json.dump(stats, f, indent=2)
    
    print(f"\n✨ Done! {len(all_models)} models saved to data/")

if __name__ == '__main__':
    main()
```

**Script**: `scripts/generate_model_pages.py`

```python
"""Generate model documentation pages from data"""
import json
import os
from pathlib import Path
from jinja2 import Template
from datetime import datetime

# Model page template
MODEL_TEMPLATE = """---
title: {{ name }}
description: {{ description }}
author: {{ author }}
tags: {{ tags }}
updated: {{ updated }}
---

# {{ name }}

{{ description }}

## Quick Start

### Using Ollama (Easiest)

```bash
# Pull and run
ollama pull {{ ollama_name }}
ollama run {{ ollama_name }}
```

### Using Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "{{ model_id }}",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{{ model_id }}")

# Generate text
inputs = tokenizer("Explain small language models:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### Using llama.cpp

```bash
# Download GGUF (quantized)
wget https://huggingface.co/{{ model_id }}/resolve/main/{{ gguf_file }}

# Run inference
./llama-cli -m {{ gguf_file }} -p "Explain SLMs:" -n 100
```

{% if colab_notebook %}
### Try in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({{ colab_notebook }})
{% endif %}

## Model Details

| Attribute | Value |
|-----------|-------|
| **Parameters** | {{ params }} |
| **Context Length** | {{ context_length }} tokens |
| **License** | {{ license }} |
| **Size** | {{ size_gb }} GB |
| **Downloads** | {{ downloads | format_number }} |
| **Likes** | {{ likes | format_number }} |
| **Released** | {{ release_date }} |

## Performance Benchmarks

{% if benchmarks %}
| Benchmark | Score |
|-----------|-------|
{% for bench, score in benchmarks.items() %}
| {{ bench }} | {{ score }} |
{% endfor %}
{% else %}
*Benchmarks not yet available*
{% endif %}

## Hardware Requirements

### Minimum

- **RAM**: {{ min_ram }} GB
- **VRAM** (FP16): {{ vram_fp16 }} GB
- **VRAM** (INT8): {{ vram_int8 }} GB
- **VRAM** (INT4): {{ vram_int4 }} GB

### Recommended

- **GPU**: NVIDIA RTX 3090 or better
- **RAM**: {{ rec_ram }} GB
- **Storage**: {{ storage }} GB

### Edge Deployment

{% if raspberry_pi_compatible %}
✅ **Raspberry Pi 5** - Runs with INT4 quantization
{% endif %}
{% if jetson_compatible %}
✅ **Jetson Nano** - Supported with optimizations
{% endif %}
{% if mobile_compatible %}
✅ **Mobile** - iOS (Core ML) / Android (TFLite)
{% endif %}

## Use Cases

Ideal for:
{% for use_case in use_cases %}
- {{ use_case }}
{% endfor %}

## Deployment Guides

- [Raspberry Pi Deployment](../deploy/hardware/raspberry-pi.md)
- [Production Deployment](../deploy/production/scaling.md)
- [Edge-First Pattern](../deploy/patterns/edge-first-cloud-fallback.md)

## Fine-Tuning

Learn how to fine-tune this model:
- [Basic Fine-Tuning](../learn/tutorials/first-fine-tune.md)
- [Advanced Techniques](../learn/tutorials/advanced/quantization-deep-dive.md)

## Resources

- [🤗 Hugging Face](https://huggingface.co/{{ model_id }})
- [📄 Model Card](https://huggingface.co/{{ model_id }}#model-card)
{% if github_repo %}
- [💻 GitHub]({{ github_repo }})
{% endif %}

## Community

- [Discussions](https://github.com/slm-hub/docs/discussions)
- [Share Your Project](../community/showcase.md)

---

*Last updated: {{ updated }} (auto-generated from Hugging Face Hub)*
"""

def format_number(n):
    """Format number with commas"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

def generate_model_page(model_data):
    """Generate markdown page for model"""
    template = Template(MODEL_TEMPLATE)
    
    # Prepare template data
    data = {
        'model_id': model_data['id'],
        'name': model_data['name'],
        'author': model_data['author'],
        'description': f"Small language model by {model_data['author']}",
        'tags': ', '.join(model_data.get('tags', [])[:5]),
        'updated': model_data['updated_at'],
        'ollama_name': model_data['name'].lower().replace(' ', '-'),
        'downloads': model_data['downloads'],
        'likes': model_data['likes'],
        'size_gb': model_data.get('size_gb', 'Unknown'),
        'params': estimate_params(model_data),
        'context_length': estimate_context(model_data),
        'license': model_data.get('card_data', {}).get('license', 'Unknown'),
        'release_date': model_data.get('created_at', 'Unknown')[:10],
        'benchmarks': model_data.get('benchmarks', {}),
        'min_ram': estimate_ram(model_data, 'min'),
        'rec_ram': estimate_ram(model_data, 'rec'),
        'vram_fp16': estimate_vram(model_data, 'fp16'),
        'vram_int8': estimate_vram(model_data, 'int8'),
        'vram_int4': estimate_vram(model_data, 'int4'),
        'storage': model_data.get('size_gb', 10),
        'raspberry_pi_compatible': model_data.get('size_gb', 999) < 3,
        'jetson_compatible': model_data.get('size_gb', 999) < 5,
        'mobile_compatible': model_data.get('size_gb', 999) < 2,
        'use_cases': generate_use_cases(model_data),
        'gguf_file': f"{model_data['name']}-Q4_K_M.gguf",
        'colab_notebook': None,  # TODO: Link to Colab
        'github_repo': None,  # TODO: Extract from card
    }
    
    # Add custom Jinja filters
    template.globals['format_number'] = format_number
    
    return template.render(**data)

def estimate_params(model_data):
    """Estimate parameter count from size"""
    size_gb = model_data.get('size_gb', 0)
    if size_gb < 2:
        return "< 2B"
    elif size_gb < 5:
        return "2-7B"
    elif size_gb < 15:
        return "7-15B"
    else:
        return "> 15B"

def estimate_context(model_data):
    """Estimate context length"""
    name = model_data['name'].lower()
    if '32k' in name or '32768' in name:
        return 32768
    elif '16k' in name or '16384' in name:
        return 16384
    elif '8k' in name or '8192' in name:
        return 8192
    return 4096

def estimate_ram(model_data, type='min'):
    """Estimate RAM requirement"""
    size_gb = model_data.get('size_gb', 0)
    multiplier = 1.5 if type == 'min' else 2.0
    return int(size_gb * multiplier)

def estimate_vram(model_data, precision):
    """Estimate VRAM requirement"""
    size_gb = model_data.get('size_gb', 0)
    multipliers = {'fp16': 1.0, 'int8': 0.5, 'int4': 0.25}
    return int(size_gb * multipliers.get(precision, 1.0))

def generate_use_cases(model_data):
    """Generate use cases based on model characteristics"""
    use_cases = []
    tags = [t.lower() for t in model_data.get('tags', [])]
    
    if 'code' in tags or 'python' in tags:
        use_cases.append("Code generation and completion")
    if 'chat' in tags or 'conversational' in tags:
        use_cases.append("Chatbots and conversational AI")
    if 'reasoning' in tags:
        use_cases.append("Complex reasoning tasks")
    if 'multilingual' in tags:
        use_cases.append("Multilingual applications")
    
    # Default use cases
    if not use_cases:
        use_cases = [
            "General text generation",
            "Question answering",
            "Content creation"
        ]
    
    return use_cases

def main():
    """Generate all model pages"""
    # Load model data
    with open('data/models.json') as f:
        models = json.load(f)
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Generate pages
    for model in models:
        page = generate_model_page(model)
        filename = f"models/{model['name'].lower().replace(' ', '-')}.md"
        
        with open(filename, 'w') as f:
            f.write(page)
        
        print(f"✅ Generated {filename}")
    
    # Generate directory index
    generate_directory_index(models)
    
    print(f"\n✨ Generated {len(models)} model pages")

def generate_directory_index(models):
    """Generate models/directory.md"""
    content = """---
title: Model Directory
description: Complete directory of Small Language Models
---

# Model Directory

*Auto-updated daily from Hugging Face Hub*

## Featured Models

"""
    
    featured = [m for m in models if m.get('featured')]
    for model in featured:
        content += f"- [{model['name']}](./{model['name'].lower().replace(' ', '-')}.md) - {model['author']}\n"
    
    content += "\n## All Models\n\n"
    content += "| Model | Author | Size | Downloads | License |\n"
    content += "|-------|--------|------|-----------|----------|\n"
    
    for model in sorted(models, key=lambda m: m['downloads'], reverse=True):
        name_link = f"[{model['name']}](./{model['name'].lower().replace(' ', '-')}.md)"
        downloads = format_number(model['downloads'])
        license = model.get('card_data', {}).get('license', 'Unknown')
        content += f"| {name_link} | {model['author']} | {model.get('size_gb', 'N/A')}GB | {downloads} | {license} |\n"
    
    with open('models/directory.md', 'w') as f:
        f.write(content)
    
    print("✅ Generated models/directory.md")

if __name__ == '__main__':
    main()
```

#### 2. Code Validation (On PR)

**File**: `.github/workflows/validate-code.yml`

```yaml
name: Validate Code Examples

on:
  pull_request:
    paths:
      - '**.md'
      - 'scripts/**'
      - 'notebooks/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install test dependencies
        run: |
          pip install pytest nbformat nbconvert black flake8
      
      - name: Extract code blocks from Markdown
        run: |
          python scripts/extract_code_blocks.py
      
      - name: Run Python code tests
        run: |
          pytest tests/code_validation/ -v
      
      -
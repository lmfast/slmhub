"""
Sync model metadata from Hugging Face Hub and generate GitBook pages.

Goals:
- Keep docs current without manual edits.
- Avoid benchmark leaderboards; focus on discovery and “how to run”.

Outputs:
- data/models.json
- models/generated/directory.mdx
- models/generated/<slug>.md

Run:
  python scripts/sync_models.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from huggingface_hub import HfApi
from jinja2 import Template


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
GENERATED_DIR = REPO_ROOT / "src" / "content" / "docs" / "docs" / "models" / "generated"

# Ensure precise path matching
if not GENERATED_DIR.exists():
    # Fallback if structure is different
    GENERATED_DIR = REPO_ROOT / "models" / "generated"

FEATURED_MODELS = [
    # Keep this list small and opinionated; it’s used for stable “featured” pages.
    # Avoid gated/private models here to keep sync reliable without credentials.
    "microsoft/Phi-4",
    "Qwen/Qwen3-8B",
    "google/gemma-2-2b-it",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]


DISCOVERY_FILTERS = [
    # Lightweight discovery: newest and popular models by task.
    # NOTE: HF filters are best-effort; we keep this minimal.
    {"task": "text-generation", "limit": 40},
    {"task": "text2text-generation", "limit": 20},
    {"task": "text-to-image", "limit": 0},  # ignored; not SLM core here
    {"task": "text-to-code", "limit": 20},
]


MODEL_PAGE_TEMPLATE = """---
title: "{{ display_name }}"
description: {{ description }}
---

# {{ display_name }}

{{ description }}

## Quickstart

### Python (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{{ model_id }}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

prompt = "Explain small language models in 3 bullets."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Local dev (Ollama)

If there’s an Ollama-friendly name for this model in your environment, you can run:

```bash
ollama run {{ ollama_hint }}
```

> If that doesn’t work, use the Transformers path above or see `deploy/quickstarts/ollama.md` for the general workflow.

## When to use this model

{% for item in when_to_use %}
- {{ item }}
{% endfor %}

## What to watch out for

{% for item in pitfalls %}
- {{ item }}
{% endfor %}

## Metadata (from Hugging Face)

| Field | Value |
|---|---|
| **Model ID** | `{{ model_id }}` |
| **Author** | {{ author }} |
| **Pipeline tag** | {{ pipeline_tag }} |
| **Library** | {{ library_name }} |
| **License** | {{ license }} |
| **Last modified** | {{ last_modified }} |
| **Downloads** | {{ downloads }} |
| **Likes** | {{ likes }} |

## Links

- [🤗 Hugging Face model page]({{ hf_url }})
- [Model card]({{ hf_url }}#model-card)

---

*This page is auto-generated from Hugging Face Hub metadata. If something looks wrong, please open a PR or issue.*
"""


DIRECTORY_TEMPLATE = """---
title: Model Directory
description: "Auto-updated model directory from Hugging Face Hub (discovery-focused)."
---

import DataTable from '../../../../../components/DataTable.astro';

This directory is **auto-updated**. It's meant for **discovery** and fast decisions, not competitive rankings.

## Featured Models

{% for m in featured -%}
- **[{{ m.display_name }}](./{{ m.slug }})** — {{ m.author }}
{% endfor %}

## Complete Model Directory

Explore all available models. Use the **search**, **filters**, and **sort** features to find what you need.

<DataTable
  columns={[
    {
      key: 'model',
      label: 'Model',
      sortable: true,
      searchable: true,
      render: (value, row) => `<a href="./${row.slug}/" class="dt-model-link">${value}</a>`
    },
    {
      key: 'author',
      label: 'Author',
      sortable: true,
      searchable: true
    },
    {
      key: 'task',
      label: 'Task',
      sortable: true,
      searchable: true
    },
    {
      key: 'license',
      label: 'License',
      sortable: true,
      searchable: true
    },
    {
      key: 'lastModified',
      label: 'Last Modified',
      sortable: true,
      searchable: false
    }
  ]}
  data={ {{ table_data }} }
  itemsPerPage={10}
  searchable={true}
  filterable={true}
  striped={true}
  hoverable={true}
  bordered={true}
/>

---

**Last sync**: {{ synced_at }}

## Tips for Finding Models

- **Search**: Use the search bar to find models by name, author, or task
- **Filter**: Use column filters to narrow down by specific criteria
- **Sort**: Click column headers to sort by that column
- **Pagination**: Navigate through results with pagination controls
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "model"


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def fmt_int(n: Optional[int]) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def fmt_size(size_bytes: Optional[int]) -> str:
    """Format model size in human-readable format."""
    if size_bytes is None or size_bytes == 0:
        return "—"
    if size_bytes >= 1_000_000_000_000:  # TB
        return f"{size_bytes / 1_000_000_000_000:.2f}TB"
    if size_bytes >= 1_000_000_000:  # GB
        return f"{size_bytes / 1_000_000_000:.2f}GB"
    if size_bytes >= 1_000_000:  # MB
        return f"{size_bytes / 1_000_000:.2f}MB"
    if size_bytes >= 1_000:  # KB
        return f"{size_bytes / 1_000:.2f}KB"
    return f"{size_bytes}B"


@dataclass(frozen=True)
class ModelDoc:
    model_id: str
    display_name: str
    slug: str
    author: str
    pipeline_tag: str
    library_name: str
    license: str
    last_modified: str
    last_modified_short: str
    downloads: str
    likes: str
    hf_url: str
    tags: List[str]
    featured: bool
    model_size: str


def extract_license(card_data: Optional[Dict[str, Any]]) -> str:
    if not card_data:
        return "unknown"
    lic = card_data.get("license")
    if isinstance(lic, str) and lic.strip():
        return lic.strip()
    return "unknown"


def model_to_doc(model_info: Any, featured: bool) -> ModelDoc:
    model_id = safe_str(getattr(model_info, "modelId", None) or getattr(model_info, "id", None))
    author = safe_str(getattr(model_info, "author", "")) or (model_id.split("/")[0] if "/" in model_id else "")
    display_name = model_id.split("/")[-1] if model_id else "model"
    slug = slugify(model_id.replace("/", "-"))

    pipeline_tag = safe_str(getattr(model_info, "pipeline_tag", "")) or "—"
    library_name = safe_str(getattr(model_info, "library_name", "")) or "—"
    downloads = fmt_int(getattr(model_info, "downloads", None))
    likes = fmt_int(getattr(model_info, "likes", None))

    last_modified_dt = getattr(model_info, "last_modified", None)
    last_modified = safe_str(last_modified_dt.isoformat() if last_modified_dt else "—")
    last_modified_short = safe_str(last_modified_dt.date().isoformat() if last_modified_dt else "—")

    card_data = getattr(model_info, "card_data", None) if hasattr(model_info, "card_data") else None
    license_name = extract_license(card_data if isinstance(card_data, dict) else None)

    tags = getattr(model_info, "tags", None)
    tags_list = [t for t in (tags or []) if isinstance(t, str)]
    
    # Extract model size from safetensors_index or files
    model_size_bytes = None
    try:
        if hasattr(model_info, "safetensors_index") and model_info.safetensors_index:
            if isinstance(model_info.safetensors_index, dict):
                model_size_bytes = sum(
                    int(meta.get("size", 0)) 
                    for meta in model_info.safetensors_index.get("metadata", {}).values()
                    if isinstance(meta, dict) and "size" in meta
                )
        # Fallback: try to get from siblings/files if available
        if not model_size_bytes and hasattr(model_info, "siblings"):
            for sibling in model_info.siblings or []:
                if hasattr(sibling, "size") and sibling.size:
                    try:
                        size_val = int(sibling.size) if isinstance(sibling.size, (int, str)) else 0
                        if model_size_bytes is None:
                            model_size_bytes = 0
                        model_size_bytes += size_val
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass  # Best effort - continue without size
    
    model_size_str = fmt_size(model_size_bytes)

    return ModelDoc(
        model_id=model_id,
        display_name=display_name,
        slug=slug,
        author=author,
        pipeline_tag=pipeline_tag,
        library_name=library_name,
        license=license_name,
        last_modified=last_modified,
        last_modified_short=last_modified_short,
        downloads=downloads,
        likes=likes,
        hf_url=f"https://huggingface.co/{model_id}",
        tags=tags_list,
        featured=featured,
        model_size=model_size_str,
    )


def heuristic_when_to_use(m: ModelDoc) -> List[str]:
    tags = {t.lower() for t in m.tags}
    items: List[str] = []

    if any("code" in t for t in tags):
        items.append("Code generation and code assistance workloads.")
    if any(t in tags for t in {"multilingual", "translation"}):
        items.append("Multilingual apps (verify your target languages with a small eval set).")
    if "reasoning" in tags:
        items.append("Tasks that benefit from structured reasoning (still validate on your domain).")
    if not items:
        items.append("General text generation and assistant-style prompts.")

    items.append("You need lower latency and lower cost than a large model.")
    return items


def heuristic_pitfalls(m: ModelDoc) -> List[str]:
    items = [
        "Treat upstream metadata as hints; validate behavior on your own prompts and data.",
        "If you see quality regressions, pin a model revision instead of tracking `main`.",
        "Quantization can change behavior; re-run acceptance tests after changing dtype/bit-width.",
    ]
    if m.license == "unknown":
        items.insert(0, "License is unknown in metadata; verify the model card before production use.")
    return items


def render_model_page(m: ModelDoc) -> str:
    t = Template(MODEL_PAGE_TEMPLATE)
    return t.render(
        model_id=m.model_id,
        display_name=m.display_name,
        description=f"Discovery-focused notes and runnable examples for `{m.model_id}`.",
        author=m.author or "—",
        pipeline_tag=m.pipeline_tag,
        library_name=m.library_name,
        license=m.license,
        last_modified=m.last_modified,
        downloads=m.downloads,
        likes=m.likes,
        hf_url=m.hf_url,
        when_to_use=heuristic_when_to_use(m),
        pitfalls=heuristic_pitfalls(m),
        ollama_hint=m.display_name.lower(),
    )


def render_directory(featured: List[ModelDoc], discovered: List[ModelDoc]) -> str:
    # Prepare data for DataTable
    all_models = featured + discovered
    table_data = []
    for m in all_models:
        table_data.append({
            "model": m.display_name,
            "author": m.author,
            "task": m.pipeline_tag,
            "license": m.license,
            "lastModified": m.last_modified_short,
            "slug": m.slug,
        })
    
    t = Template(DIRECTORY_TEMPLATE)
    return t.render(
        featured=featured, 
        discovered=discovered, 
        synced_at=utc_now_iso(),
        table_data=json.dumps(table_data)
    )


def unique_by_model_id(models: Iterable[ModelDoc]) -> List[ModelDoc]:
    seen: set[str] = set()
    out: List[ModelDoc] = []
    for m in models:
        if m.model_id in seen:
            continue
        seen.add(m.model_id)
        out.append(m)
    return out


def main() -> None:
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    featured_docs: List[ModelDoc] = []
    for model_id in FEATURED_MODELS:
        try:
            info = api.model_info(model_id)
        except Exception as exc:  # noqa: BLE001 - skip but continue
            print(f"⚠️  Skipping featured model '{model_id}': {exc}")
            continue
        featured_docs.append(model_to_doc(info, featured=True))

    discovered_docs: List[ModelDoc] = []
    for f in DISCOVERY_FILTERS:
        task = f.get("task")
        limit = int(f.get("limit", 0))
        if not task or limit <= 0:
            continue
        for m in api.list_models(filter=task, sort="lastModified", direction=-1, limit=limit):
            try:
                info = api.model_info(m.modelId)
            except Exception as exc:  # noqa: BLE001 - discovery is best-effort
                print(f"⚠️  Skipping discovered model '{m.modelId}': {exc}")
                continue
            discovered_docs.append(model_to_doc(info, featured=False))

    all_docs = unique_by_model_id(featured_docs + discovered_docs)

    # Write machine-readable data
    last_synced_at = utc_now_iso()
    data_payload: List[Dict[str, Any]] = [
        {
            "model_id": d.model_id,
            "display_name": d.display_name,
            "slug": d.slug,
            "author": d.author,
            "pipeline_tag": d.pipeline_tag,
            "library_name": d.library_name,
            "license": d.license,
            "last_modified": d.last_modified,
            "downloads": d.downloads,
            "likes": d.likes,
            "hf_url": d.hf_url,
            "tags": d.tags,
            "featured": d.featured,
            "model_size": d.model_size,
        }
        for d in all_docs
    ]
    
    # Add metadata about the sync
    output_data = {
        "last_synced_at": last_synced_at,
        "models": data_payload,
    }
    
    (DATA_DIR / "models.json").write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    (DATA_DIR / "models.yaml").write_text(yaml.safe_dump(output_data, sort_keys=False), encoding="utf-8")

    # Generate pages
    for d in all_docs:
        (GENERATED_DIR / f"{d.slug}.md").write_text(render_model_page(d), encoding="utf-8")

    # Generate directory index
    featured_sorted = sorted(featured_docs, key=lambda x: x.display_name.lower())
    discovered_sorted = sorted(
        [d for d in all_docs if not d.featured],
        key=lambda x: (x.last_modified_short, x.display_name.lower()),
        reverse=True,
    )[:200]
    (GENERATED_DIR / "directory.mdx").write_text(render_directory(featured_sorted, discovered_sorted), encoding="utf-8")

    print(f"✅ Synced {len(all_docs)} models")
    print(f"✅ Wrote: {DATA_DIR / 'models.json'}")
    print(f"✅ Wrote: {GENERATED_DIR / 'directory.mdx'}")


if __name__ == "__main__":
    main()

"""
Sync model metadata from Hugging Face Hub and generate GitBook pages.

Goals:
- Keep docs current without manual edits.
- Avoid benchmark leaderboards; focus on discovery and “how to run”.

Outputs:
- data/models.json
- data/models.yaml

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


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

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

GGUF_QUANT_TABLE: Dict[str, Dict[str, Any]] = {
    "Q2_K": {"bpw": 2.5, "quality": "Low"},
    "Q3_K_M": {"bpw": 3.4, "quality": "Medium-low"},
    "Q4_0": {"bpw": 4.5, "quality": "Balanced (legacy)"},
    "Q4_K_M": {"bpw": 4.5, "quality": "Balanced (recommended)"},
    "Q5_0": {"bpw": 5.5, "quality": "Good (legacy)"},
    "Q5_K_M": {"bpw": 5.5, "quality": "Good"},
    "Q6_K": {"bpw": 6.5, "quality": "High"},
    "Q8_0": {"bpw": 8.0, "quality": "Very high"},
    "F16": {"bpw": 16.0, "quality": "Original precision"},
}


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
    downloads_raw: Optional[int]
    likes_raw: Optional[int]


def extract_license(card_data: Optional[Dict[str, Any]]) -> str:
    if not card_data:
        return "unknown"
    lic = card_data.get("license")
    if isinstance(lic, str) and lic.strip():
        return lic.strip()
    return "unknown"


def extract_license_from_tags(tags: List[str]) -> Optional[str]:
    for t in tags:
        if not isinstance(t, str):
            continue
        if t.startswith("license:"):
            v = t.split("license:", 1)[1].strip()
            if v:
                return v
    return None

def extract_base_model(tags: List[str], card_data: Optional[Dict[str, Any]]) -> Optional[str]:
    # Prefer structured card_data if present
    if isinstance(card_data, dict):
        for key in ("base_model", "base_model_name", "base_model_id", "base_model_name_or_path"):
            v = card_data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Some cards store base_model in "model-index" etc; skip heavy parsing here.
    # Fallback: tags include base_model:...
    for t in tags:
        if not isinstance(t, str):
            continue
        if t.startswith("base_model:"):
            bm = t.split("base_model:", 1)[1].strip()
            if bm:
                # strip modifiers like "finetune:" / "quantized:"
                bm = bm.replace("finetune:", "").replace("quantized:", "").replace("adapter:", "")
                return bm.strip() or None
    return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def human_parameters(n: Optional[int]) -> Optional[str]:
    if not n:
        return None
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B".replace(".0B", "B")
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n/1_000:.1f}K".replace(".0K", "K")
    return str(n)


def approx_param_count_from_config(cfg: Dict[str, Any]) -> Optional[int]:
    """
    Best-effort parameter estimate.
    Many configs don't include a definitive count; we approximate the transformer weights.
    """
    if not isinstance(cfg, dict):
        return None
    # If explicit field exists, trust it
    for key in ("num_parameters", "n_parameters", "parameter_count"):
        v = safe_int(cfg.get(key))
        if v and v > 0:
            return v
    # Common fields
    hidden = safe_int(cfg.get("hidden_size")) or safe_int(cfg.get("d_model"))
    layers = safe_int(cfg.get("num_hidden_layers")) or safe_int(cfg.get("n_layer")) or safe_int(cfg.get("num_layers"))
    vocab = safe_int(cfg.get("vocab_size"))
    intermediate = safe_int(cfg.get("intermediate_size")) or safe_int(cfg.get("ffn_dim"))
    if not hidden or not layers:
        return None
    if not intermediate:
        intermediate = hidden * 4
    # Very rough: embeddings + per-layer (attn + ffn) weights.
    # This is approximate but consistent enough for VRAM sizing.
    embed = (vocab or 0) * hidden
    per_layer = (4 * hidden * hidden) + (2 * hidden * intermediate)  # qkv+o + ffn
    return embed + layers * per_layer


def vram_from_params(param_count: Optional[int], bpw: float, overhead: float = 1.2) -> Optional[float]:
    if not param_count:
        return None
    bytes_per_param = bpw / 8.0
    gb = (param_count * bytes_per_param * overhead) / 1_000_000_000
    return round(gb, 2)


def recommend_gpus(recommended_vram_gb: Optional[float]) -> List[str]:
    if recommended_vram_gb is None:
        return []
    v = recommended_vram_gb
    if v <= 4:
        return ["RTX 3050", "GTX 1650"]
    if v <= 8:
        return ["RTX 3060 (8GB)", "RTX 4060"]
    if v <= 12:
        return ["RTX 3060 (12GB)", "RTX 4060 Ti"]
    if v <= 16:
        return ["RTX 4070", "RTX 3080"]
    if v <= 24:
        return ["RTX 4080", "RTX 4090"]
    return ["A6000", "H100"]


def get_repo_files_with_sizes(model_info: Any) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {}
    siblings = getattr(model_info, "siblings", None) or []
    for s in siblings:
        rfilename = getattr(s, "rfilename", None)
        if not rfilename:
            continue
        out[str(rfilename)] = safe_int(getattr(s, "size", None))
    return out


def parse_gguf_quant_level(filename: str) -> Optional[str]:
    # common patterns include: *Q4_K_M*.gguf, *Q5_0*.gguf, *F16*.gguf
    # keep this conservative and table-driven
    upper = filename.upper()
    for q in GGUF_QUANT_TABLE.keys():
        if q in upper:
            return q
    # try a fallback like "Q4_K_M" or "Q8_0" with punctuation normalized
    m = re.search(r"(Q[0-9]_[A-Z0-9_]+|Q[0-9]_[0-9]|Q[0-9]0|Q[0-9])", upper)
    if m:
        return m.group(1)
    return None


def extract_description_from_card_data(card_data: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(card_data, dict):
        return None
    # Prefer short summaries if present
    for key in ("summary", "model_summary", "description"):
        v = card_data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def extract_description_from_readme(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    # Strip YAML frontmatter if present
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2]
    # Remove leading headings and empty lines, then take the first paragraph-like chunk
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned: List[str] = []
    for ln in lines:
        if not ln:
            if cleaned:
                break
            continue
        if ln.startswith("#") and not cleaned:
            continue
        cleaned.append(ln)
    paragraph = " ".join(cleaned).strip()
    if paragraph:
        return paragraph[:500]
    return None


def fetch_json_file(api: HfApi, repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
    try:
        # This returns parsed dict when possible
        return api.hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")  # type: ignore[return-value]
    except Exception:
        return None


def load_json_path(p: Any) -> Optional[Dict[str, Any]]:
    try:
        if not p:
            return None
        path_obj = Path(str(p))
        if not path_obj.exists():
            return None
        return json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_subset(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        if k in d and d[k] is not None:
            out[k] = d[k]
    return out


def model_to_doc(model_info: Any, featured: bool) -> ModelDoc:
    model_id = safe_str(getattr(model_info, "modelId", None) or getattr(model_info, "id", None))
    author = safe_str(getattr(model_info, "author", "")) or (model_id.split("/")[0] if "/" in model_id else "")
    display_name = model_id.split("/")[-1] if model_id else "model"
    slug = slugify(model_id.replace("/", "-"))

    pipeline_tag = safe_str(getattr(model_info, "pipeline_tag", "")) or "—"
    library_name = safe_str(getattr(model_info, "library_name", "")) or "—"
    downloads_raw = safe_int(getattr(model_info, "downloads", None))
    likes_raw = safe_int(getattr(model_info, "likes", None))
    downloads = fmt_int(downloads_raw)
    likes = fmt_int(likes_raw)

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
        downloads_raw=downloads_raw,
        likes_raw=likes_raw,
    )

def derive_languages_from_tags(tags: List[str]) -> List[str]:
    langs: List[str] = []
    for t in tags:
        if not isinstance(t, str):
            continue
        # HF often uses ISO codes as tags (en, fr, zh, ...)
        if re.fullmatch(r"[a-z]{2}", t):
            langs.append(t)
    # Deduplicate preserving order
    out: List[str] = []
    seen: set[str] = set()
    for l in langs:
        if l in seen:
            continue
        seen.add(l)
        out.append(l)
    return out


def classify_primary_use_case(pipeline_tag: str, tags: List[str]) -> str:
    tagset = {t.lower() for t in tags if isinstance(t, str)}
    if "text-to-code" in tagset or "code" in tagset or any("code" in t for t in tagset):
        return "Code Generation"
    if pipeline_tag in {"translation"}:
        return "Translation"
    if pipeline_tag in {"summarization"}:
        return "Content Creation"
    if "rag" in tagset or "retrieval" in tagset:
        return "Question Answering"
    if "classification" in tagset:
        return "Classification"
    if "conversational" in tagset or "chat" in tagset:
        return "Conversational AI"
    return "Content Creation" if pipeline_tag else "Conversational AI"


def build_hardware_requirements(param_count: Optional[int]) -> Dict[str, Any]:
    fp16 = vram_from_params(param_count, bpw=16.0)
    q8 = vram_from_params(param_count, bpw=8.0)
    q4 = vram_from_params(param_count, bpw=4.5)
    rec = q4 or q8 or fp16
    return {
        "quantization_level": "Q4_K_M",
        "min_vram_gb": q4,
        "recommended_vram_gb": rec,
        "min_ram_gb": (q4 * 2) if q4 else None,
        "recommended_ram_gb": (rec * 2) if rec else None,
        "cpu_only_viable": True if (q4 and q4 <= 12) else None,
        "gpu_recommendations": recommend_gpus(rec),
        "edge_compatible": True if (q4 and q4 <= 4) else None,
        "mobile_compatible": True if (q4 and q4 <= 2) else None,
    }


def build_gguf_variants_for_model(repo_id: str, file_sizes: Dict[str, Optional[int]], param_count: Optional[int]) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for fname, size in file_sizes.items():
        if not isinstance(fname, str) or not fname.lower().endswith(".gguf"):
            continue
        q = parse_gguf_quant_level(fname)
        q_meta = GGUF_QUANT_TABLE.get(q or "", {})
        bpw = q_meta.get("bpw")
        quality = q_meta.get("quality")
        size_gb = (size / 1_000_000_000) if isinstance(size, int) and size else None
        rec_vram = vram_from_params(param_count, bpw=float(bpw)) if bpw else None
        variants.append(
            {
                "variant_repo_id": repo_id,
                "quantization_method": "gguf",
                "quantization_level": q,
                "file_name": fname,
                "file_size_bytes": size,
                "file_size_gb": round(size_gb, 3) if size_gb else None,
                "bits_per_weight": bpw,
                "estimated_quality": quality,
                "recommended_vram_gb": rec_vram,
                "variant_url": f"https://huggingface.co/{repo_id}/resolve/main/{fname}",
            }
        )
    # Prefer smaller-to-larger by bpw then size
    variants.sort(key=lambda v: ((v.get("bits_per_weight") or 999), (v.get("file_size_bytes") or 0)))
    return variants


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
    data_payload: List[Dict[str, Any]] = []

    # Build a first pass of "base models" so we can attach GGUF variants from GGUF repos
    base_to_variants: Dict[str, List[Dict[str, Any]]] = {}

    for d in all_docs:
        try:
            info = api.model_info(d.model_id)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Skipping enrichment for '{d.model_id}': {exc}")
            info = None

        card_data = getattr(info, "card_data", None) if info is not None else None
        card_data_dict = card_data if isinstance(card_data, dict) else None
        base_model = extract_base_model(d.tags, card_data_dict)
        license_from_tags = extract_license_from_tags(d.tags)
        description = extract_description_from_card_data(card_data_dict)

        file_sizes = get_repo_files_with_sizes(info) if info is not None else {}

        # Load config/tokenizer files (best-effort). Use hf_hub_download (cached) then read.
        config_path = None
        tok_cfg_path = None
        readme_path = None
        tok_json_path = None
        try:
            config_path = api.hf_hub_download(repo_id=d.model_id, filename="config.json", repo_type="model")
        except Exception:
            config_path = None
        try:
            tok_cfg_path = api.hf_hub_download(repo_id=d.model_id, filename="tokenizer_config.json", repo_type="model")
        except Exception:
            tok_cfg_path = None
        try:
            readme_path = api.hf_hub_download(repo_id=d.model_id, filename="README.md", repo_type="model")
        except Exception:
            readme_path = None
        # Only attempt tokenizer.json for smaller files (it can be huge)
        tok_json_size = file_sizes.get("tokenizer.json")
        if isinstance(tok_json_size, int) and 0 < tok_json_size <= 5_000_000:
            try:
                tok_json_path = api.hf_hub_download(repo_id=d.model_id, filename="tokenizer.json", repo_type="model")
            except Exception:
                tok_json_path = None

        cfg = load_json_path(config_path) or {}
        tok_cfg = load_json_path(tok_cfg_path) or {}
        tok_json = load_json_path(tok_json_path) or {}

        if not description and readme_path:
            try:
                readme_txt = Path(str(readme_path)).read_text(encoding="utf-8", errors="ignore")
                description = extract_description_from_readme(readme_txt) or description
            except Exception:
                pass

        architecture = None
        if isinstance(cfg, dict):
            architecture = safe_str(cfg.get("model_type") or cfg.get("architectures") or cfg.get("architectures", None))
            if isinstance(cfg.get("model_type"), str):
                architecture = cfg.get("model_type")
        if architecture and isinstance(architecture, list):
            architecture = safe_str(architecture[0] if architecture else None)

        param_count = approx_param_count_from_config(cfg if isinstance(cfg, dict) else {})
        params_h = human_parameters(param_count)

        tech_spec = safe_subset(
            cfg if isinstance(cfg, dict) else {},
            [
                "vocab_size",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "max_position_embeddings",
                "intermediate_size",
                "hidden_act",
                "rope_theta",
                "sliding_window",
                "rms_norm_eps",
                "tie_word_embeddings",
            ],
        )
        # Normalize a few keys to match requested schema names
        technical_specifications = {
            "vocab_size": tech_spec.get("vocab_size"),
            "hidden_size": tech_spec.get("hidden_size"),
            "num_layers": tech_spec.get("num_hidden_layers"),
            "num_attention_heads": tech_spec.get("num_attention_heads"),
            "num_key_value_heads": tech_spec.get("num_key_value_heads"),
            "context_length": tech_spec.get("max_position_embeddings"),
            "intermediate_size": tech_spec.get("intermediate_size"),
            "activation_function": tech_spec.get("hidden_act"),
            "rope_theta": tech_spec.get("rope_theta"),
            "sliding_window": tech_spec.get("sliding_window"),
            "rms_norm_eps": tech_spec.get("rms_norm_eps"),
            "tie_word_embeddings": tech_spec.get("tie_word_embeddings"),
        }
        technical_specifications = {k: v for k, v in technical_specifications.items() if v is not None} or None

        tokenizer_information = None
        if isinstance(tok_cfg, dict) and tok_cfg:
            vocab_size_tok = None
            if isinstance(tok_json, dict):
                # tokenizers JSON: {"model": {"vocab": {...}}} or {"model": {"vocab": [...]}}
                model_obj = tok_json.get("model")
                if isinstance(model_obj, dict):
                    vocab = model_obj.get("vocab")
                    if isinstance(vocab, dict):
                        vocab_size_tok = len(vocab)
                    elif isinstance(vocab, list):
                        vocab_size_tok = len(vocab)
            tokenizer_information = {
                "tokenizer_type": tok_cfg.get("tokenizer_type"),
                "tokenizer_class": tok_cfg.get("tokenizer_class"),
                "vocab_size": vocab_size_tok,
                "bos_token": tok_cfg.get("bos_token"),
                "eos_token": tok_cfg.get("eos_token"),
                "pad_token": tok_cfg.get("pad_token"),
                "unk_token": tok_cfg.get("unk_token"),
                "special_tokens": tok_cfg.get("special_tokens_map") or tok_cfg.get("additional_special_tokens"),
                "model_max_length": tok_cfg.get("model_max_length"),
                "clean_up_tokenization_spaces": tok_cfg.get("clean_up_tokenization_spaces"),
                "add_bos_token": tok_cfg.get("add_bos_token"),
                "add_eos_token": tok_cfg.get("add_eos_token"),
            }
            tokenizer_information = {k: v for k, v in tokenizer_information.items() if v is not None} or None

        gguf_variants = build_gguf_variants_for_model(d.model_id, file_sizes, param_count)

        # If this repo is itself a GGUF repo and it declares a base_model, attach variants to that base
        if base_model and any(isinstance(t, str) and t.lower() == "gguf" for t in d.tags) and gguf_variants:
            base_to_variants.setdefault(base_model, []).extend(gguf_variants)

        use_cases = {
            "primary_use_case": classify_primary_use_case(d.pipeline_tag or "", d.tags),
            "task_types": [d.pipeline_tag] if d.pipeline_tag and d.pipeline_tag != "—" else [],
            "use_case_tags": [],
            "example_prompts": [],
            "limitations": None,
            "bias_considerations": None,
            "languages": derive_languages_from_tags(d.tags),
            "multimodal": any(isinstance(t, str) and t.lower() in {"vision", "audio", "multimodal"} for t in d.tags),
        }

        training_information = {
            "training_datasets": [],
            "training_tokens": None,
            "training_framework": None,
            "finetuning_method": None,
            "training_date": None,
            "base_model_name": base_model,
        }

        payload = {
            # Main models table
            "model_id": d.model_id,
            "model_name": d.display_name,
            "provider": d.author or None,
            "last_updated": d.last_modified if d.last_modified != "—" else None,
            "description": description or f"Auto-synced metadata and usage notes for `{d.model_id}`.",
            "base_model": base_model,
            "architecture": architecture,
            "parameters": params_h,
            "parameter_count": param_count,
            "likes": d.likes_raw,
            "downloads": d.downloads_raw,
            "downloads_30d": None,
            "license": license_from_tags or d.license,
            "library": d.library_name if d.library_name != "—" else None,
            "pipeline_tag": d.pipeline_tag if d.pipeline_tag != "—" else None,
            "model_url": d.hf_url,
            # UI/back-compat fields used by directory.astro
            "display_name": d.display_name,
            "author": d.author,
            "slug": d.slug,
            "likes_human": d.likes,
            "downloads_human": d.downloads,
            "model_size": d.model_size,
            "tags": d.tags,
            # Detail sections
            "technical_specifications": technical_specifications,
            "tokenizer_information": tokenizer_information,
            "gguf_variants": gguf_variants,
            "hardware_requirements": build_hardware_requirements(param_count),
            "use_cases": use_cases,
            "training_information": training_information,
        }
        data_payload.append(payload)

    # Second pass: merge GGUF variants discovered in GGUF repos into their base models
    for m in data_payload:
        mid = m.get("model_id")
        if not isinstance(mid, str):
            continue
        extra = base_to_variants.get(mid) or []
        if not extra:
            continue
        existing = m.get("gguf_variants") or []
        if isinstance(existing, list):
            # dedupe on repo+file
            seen = {(v.get("variant_repo_id"), v.get("file_name")) for v in existing if isinstance(v, dict)}
            for v in extra:
                key = (v.get("variant_repo_id"), v.get("file_name"))
                if key in seen:
                    continue
                seen.add(key)
                existing.append(v)
            m["gguf_variants"] = existing
    
    # Add metadata about the sync
    output_data = {
        "last_synced_at": last_synced_at,
        "models": data_payload,
    }
    
    (DATA_DIR / "models.json").write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    (DATA_DIR / "models.yaml").write_text(yaml.safe_dump(output_data, sort_keys=False), encoding="utf-8")

    print(f"✅ Synced {len(all_docs)} models")
    print(f"✅ Wrote: {DATA_DIR / 'models.json'}")
    print(f"✅ Wrote: {DATA_DIR / 'models.yaml'}")


if __name__ == "__main__":
    main()

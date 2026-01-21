# SLM Hub: Complete Course Notebooks

**The Developer's Guide to Small Language Models (2026)**

All notebooks are designed to run on **Google Colab T4 (free tier)** with executable code, visualizations, and hands-on exercises.

---

## 📚 **SECTION 1: FOUNDATIONS**

### Module 1.1: Neural Networks - The Basics (60 mins)
**Notebook**: [`foundations/01_neural_networks_basics.ipynb`](./foundations/01_neural_networks_basics.ipynb)

**What you'll learn**:
- ✅ Build a 2-layer neural network from scratch
- ✅ Visualize forward/backward passes
- ✅ Implement single-head attention
- ✅ Scale to multi-head attention (8 heads)
- ✅ Visualize attention heatmaps

**Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmfast/slmhub/blob/main/notebooks/foundations/01_neural_networks_basics.ipynb)

---

### Module 1.2: Transformer Architecture Deep Dive (50 mins)
**Notebook**: [`foundations/02_transformer_architecture.ipynb`](./foundations/02_transformer_architecture.ipynb)

**What you'll learn**:
- Position encoding (sinusoidal)
- RoPE (Rotary Position Embedding)
- Feed-forward networks with SwiGLU
- Layer normalization vs RMSNorm
- Complete transformer block

**Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmfast/slmhub/blob/main/notebooks/foundations/02_transformer_architecture.ipynb)

---

### Module 1.3-1.4: Tokenization & Complete Transformer (75 mins)
**Notebook**: [`foundations/03_tokenization_transformer.ipynb`](./foundations/03_tokenization_transformer.ipynb)

**What you'll learn**:
- BPE tokenizer from scratch
- SentencePiece comparison
- Build complete SmolLM-135M architecture
- Forward pass with real tokens

---

### Module 1.5-1.6: KV Cache & Optimization (85 mins)
**Notebook**: [`foundations/04_kv_cache_optimization.ipynb`](./foundations/04_kv_cache_optimization.ipynb)

**What you'll learn**:
- Why autoregressive generation is slow (O(n²))
- Implement KV cache (10x speedup)
- Grouped Query Attention (GQA)
- Multi-Head Latent Attention (MLA)

---

### Module 1.7: Hardware & GPU Basics (30 mins)
**Notebook**: [`foundations/05_hardware_gpu_basics.ipynb`](./foundations/05_hardware_gpu_basics.ipynb)

**What you'll learn**:
- GPU taxonomy (T4, RTX 4090, A100, H100)
- FP32 vs FP16 vs BF16 vs INT8 vs INT4
- Memory calculations
- Interactive GPU picker tool

---

### Module 1.8: Quantization Methods (60 mins)
**Notebook**: [`foundations/06_quantization_methods.ipynb`](./foundations/06_quantization_methods.ipynb)

**What you'll learn**:
- GPTQ (layer-wise quantization)
- AWQ (activation-aware)
- GGUF for CPU inference
- Benchmark: perplexity, speed, memory

---

### Module 1.9: Training Optimizations (75 mins)
**Notebook**: [`foundations/07_training_optimizations.ipynb`](./foundations/07_training_optimizations.ipynb)

**What you'll learn**:
- Gradient accumulation
- Gradient checkpointing
- Mixed precision training (AMP)
- Memory vs compute trade-offs

---

### Module 1.10: Scaling Laws (30 mins)
**Notebook**: [`foundations/08_scaling_laws.ipynb`](./foundations/08_scaling_laws.ipynb)

**What you'll learn**:
- Chinchilla scaling formula
- Compute budget calculator
- Training time estimation
- Interactive calculator

---

## 📚 **SECTION 2: MODELS**

### Module 2.1: SLM Model Zoo (45 mins)
**Notebook**: [`models/01_model_zoo.ipynb`](./models/01_model_zoo.ipynb)

**Featured Models**:
- SmolLM Family (135M, 360M, 1.7B)
- Phi-3 Family
- Qwen2.5 Family
- Gemma-2, MiniCPM, StableLM-2
- Interactive comparison tool

---

### Module 2.2: Benchmarking Deep Dive (100 mins)
**Notebook**: [`models/02_benchmarking.ipynb`](./models/02_benchmarking.ipynb)

**What you'll learn**:
- Perplexity, MMLU, HumanEval, GSM8K, HellaSwag
- Run your own benchmarks
- Visualize results (radar charts)

---

### Module 2.3: Domain-Specific Models (60 mins)
**Notebook**: [`models/03_domain_specific.ipynb`](./models/03_domain_specific.ipynb)

**Domains**:
- Code (StarCoder2, CodeQwen, DeepSeek-Coder)
- Math (DeepSeek-Math, MathGPT)
- Multilingual (XGLM, mGPT)

---

### Module 2.4: Hardware Requirements Matrix (30 mins)
**Notebook**: [`models/04_hardware_requirements.ipynb`](./models/04_hardware_requirements.ipynb)

**What you'll learn**:
- Memory requirements per precision
- Batch size calculations
- Interactive calculator

---

## 📚 **SECTION 3: HANDS-ON COURSE**

### Module 3.1: Your First SLM in 10 Minutes ⚡
**Notebook**: [`hands_on/01_first_slm_10min.ipynb`](./hands_on/01_first_slm_10min.ipynb)

**What you'll build**:
- Load SmolLM-135M
- Generate text
- Experiment with temperature, top_k, top_p

**Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmfast/slmhub/blob/main/notebooks/hands_on/01_first_slm_10min.ipynb)

---

### Module 3.2: Fine-Tuning with LoRA (90 mins)
**Notebook**: [`hands_on/02_fine_tuning_lora.ipynb`](./hands_on/02_fine_tuning_lora.ipynb)

**What you'll learn**:
- LoRA configuration (r, alpha, target modules)
- QLoRA for extreme efficiency (<2GB on T4!)
- Training loop
- Merge and export

---

### Module 3.3: DPO (Direct Preference Optimization) (60 mins)
**Notebook**: [`hands_on/03_dpo_alignment.ipynb`](./hands_on/03_dpo_alignment.ipynb)

**What you'll learn**:
- Align model to human preferences
- Chosen vs rejected responses
- KL penalty tuning

---

### Module 3.4: Function Calling & Tool Use (75 mins)
**Notebook**: [`hands_on/04_function_calling_agents.ipynb`](./hands_on/04_function_calling_agents.ipynb)

**What you'll build**:
- Multi-tool agent (weather, calculator, search, Wikipedia)
- ReAct pattern implementation
- FunctionGemma integration

---

### Module 3.5: RAG (Retrieval-Augmented Generation) (90 mins)
**Notebook**: [`hands_on/05_rag_system.ipynb`](./hands_on/05_rag_system.ipynb)

**What you'll build**:
- Complete RAG pipeline
- Chunking, embeddings, vector store (FAISS)
- Query and generate
- Advanced: GraphRAG, HyDE, Self-RAG

---

### Module 3.6: Inference Optimization (75 mins)
**Notebook**: [`hands_on/06_inference_optimization.ipynb`](./hands_on/06_inference_optimization.ipynb)

**What you'll learn**:
- Speculative decoding (2-3x speedup)
- FlashAttention
- Kernel optimization

---

### Module 3.7: Deployment Strategies (150 mins)
**Notebook**: [`hands_on/07_deployment.ipynb`](./hands_on/07_deployment.ipynb)

**What you'll deploy**:
- vLLM serving (1000+ tokens/sec)
- Mobile (ONNX Runtime)
- Browser (WebGPU)

---

### Module 3.8: Prompt Engineering (45 mins)
**Notebook**: [`hands_on/08_prompt_engineering.ipynb`](./hands_on/08_prompt_engineering.ipynb)

**Techniques**:
- Few-shot learning
- Chain-of-thought
- Role prompting

---

### Module 3.9: Evaluation & Monitoring (40 mins)
**Notebook**: [`hands_on/09_evaluation_monitoring.ipynb`](./hands_on/09_evaluation_monitoring.ipynb)

**What you'll learn**:
- Production metrics (latency, throughput, error rate)
- Grafana + Prometheus setup
- A/B testing

---

### Module 3.10: Dataset Engineering (60 mins)
**Notebook**: [`hands_on/10_dataset_engineering.ipynb`](./hands_on/10_dataset_engineering.ipynb)

**What you'll learn**:
- Data sources (FineWeb-Edu, The Stack, GSM8K)
- Quality filtering
- Deduplication

---

## 🚀 **Quick Start**

### Option 1: Run Locally
```bash
# Clone repository
git clone https://github.com/lmfast/slmhub.git
cd slmhub/notebooks

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Option 2: Google Colab (Recommended)
Click any "Open in Colab" badge above to run in your browser with free GPU!

---

## 📊 **Learning Path**

### Beginner Track (10 hours)
1. Module 1.1: Neural Networks Basics
2. Module 3.1: Your First SLM in 10 Minutes
3. Module 3.8: Prompt Engineering
4. Module 2.1: Model Zoo

### Intermediate Track (25 hours)
1. Complete Section 1 (Foundations)
2. Module 3.2: Fine-Tuning with LoRA
3. Module 3.5: RAG System
4. Module 2.2: Benchmarking

### Advanced Track (40+ hours)
1. All modules in order
2. Complete all exercises
3. Build custom projects

---

## 🛠️ **Requirements**

### Minimum
- Python 3.8+
- 8GB RAM
- CPU (slow but functional)

### Recommended
- Python 3.10+
- 16GB RAM
- GPU with 8GB+ VRAM (or Google Colab T4)

### Dependencies
```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
peft>=0.6.0
trl>=0.7.0
bitsandbytes>=0.41.0
sentencepiece>=0.1.99
tokenizers>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
pandas>=2.0.0
```

---

## 📖 **Additional Resources**

- **Documentation**: [slmhub.dev](https://lmfast.github.io/slmhub)
- **Discord**: [Join Community](https://discord.gg/slmhub)
- **GitHub**: [lmfast/slmhub](https://github.com/lmfast/slmhub)
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## 📝 **License**

Apache 2.0 - See [LICENSE](../LICENSE)

---

**Built with ❤️ by the SLM Hub community**

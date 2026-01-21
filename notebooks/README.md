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

### Module 1.3: Feed-Forward Networks & Normalization (45 mins)
**Notebook**: [`foundations/03_feedforward_normalization.ipynb`](./foundations/03_feedforward_normalization.ipynb)

**What you'll learn**:
- FFN with SwiGLU activation
- Compare ReLU vs GELU vs SwiGLU
- LayerNorm vs RMSNorm implementation
- Memory profiling

---

### Module 1.4: Complete Transformer Block (60 mins)
**Notebook**: [`foundations/04_complete_transformer_block.ipynb`](./foundations/04_complete_transformer_block.ipynb)

**What you'll learn**:
- Build complete SmolLM-135M transformer layer
- Architecture: 9 layers, 576 hidden, 9 heads, 1536 FFN
- Forward pass with real tokens
- Component profiling (time/memory)

---

### Module 1.5: Tokenization (50 mins)
**Notebook**: [`foundations/05_tokenization.ipynb`](./foundations/05_tokenization.ipynb)

**What you'll learn**:
- BPE tokenizer from scratch
- Train on Wikipedia sample
- SentencePiece comparison
- Vocabulary size trade-offs (10K vs 50K vs 100K)
- Multilingual tokenization

---

### Module 1.6: KV Cache & Attention Optimization (50 mins)
**Notebook**: [`foundations/06_kv_cache.ipynb`](./foundations/06_kv_cache.ipynb)

**What you'll learn**:
- Naive generation (O(n²) problem)
- KV cache implementation
- GQA (Grouped Query Attention)
- MLA (Multi-Head Latent Attention)
- Benchmark: MHA vs GQA vs MLA

---

### Module 1.7: Hardware & GPU Basics (40 mins)
**Notebook**: [`foundations/07_hardware_gpu_basics.ipynb`](./foundations/07_hardware_gpu_basics.ipynb)

**What you'll learn**:
- GPU taxonomy (T4, RTX 4090, A100, H100)
- FP32 vs FP16 vs BF16 vs INT8 vs INT4
- Memory calculations
- Interactive GPU picker tool

---

### Module 1.8: Quantization Methods (60 mins)
**Notebook**: [`foundations/08_quantization_methods.ipynb`](./foundations/08_quantization_methods.ipynb)

**What you'll learn**:
- GPTQ implementation
- AWQ implementation
- GGUF for CPU
- Benchmark suite: perplexity, MMLU, speed, memory
- Quantize SmolLM-135M with all methods

---

### Module 1.9: Training Optimizations (45 mins)
**Notebook**: [`foundations/09_training_optimizations.ipynb`](./foundations/09_training_optimizations.ipynb)

**What you'll learn**:
- Gradient accumulation
- Gradient checkpointing
- Mixed precision training (AMP)
- Memory vs compute trade-offs

---

### Module 1.10: Scaling Laws (40 mins)
**Notebook**: [`foundations/10_scaling_laws.ipynb`](./foundations/10_scaling_laws.ipynb)

**What you'll learn**:
- Chinchilla scaling formula
- Compute budget calculator
- Training time estimation
- Interactive calculator

---

## 📚 **SECTION 2: MODELS**

### Module 2.1: Model Zoo (60 mins)
**Notebook**: [`models/01_model_zoo.ipynb`](./models/01_model_zoo.ipynb)

**Featured Models**:
- SmolLM Family (135M, 360M, 1.7B)
- Phi-3 Family
- Qwen2.5 Family
- Gemma-2, MiniCPM, StableLM-2
- Interactive comparison tool

---

### Module 2.2: Benchmarking (90 mins)
**Notebook**: [`models/02_benchmarking.ipynb`](./models/02_benchmarking.ipynb)

**What you'll learn**:
- Perplexity, MMLU, HumanEval, GSM8K, HellaSwag
- Run your own benchmarks
- Visualize results (radar charts)

---

### Module 2.3: Domain-Specific Models (50 mins)
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

## 📚 **SECTION 4: ADVANCED TOPICS**

### Module 4.1: Mixture of Experts (MoE) (90 mins)
**Notebook**: [`advanced_topics/01_mixture_of_experts.ipynb`](./advanced_topics/01_mixture_of_experts.ipynb)

**What you'll learn**:
- ✅ MoE layer implementation from scratch
- ✅ Router mechanism with top-k selection
- ✅ Compare dense vs MoE memory/compute
- ✅ Convert existing model to MoE
- ✅ Visualize expert routing patterns

**Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmfast/slmhub/blob/main/notebooks/advanced_topics/01_mixture_of_experts.ipynb)

---

### Module 4.2: Sliding Window Attention (75 mins)
**Notebook**: [`advanced_topics/02_sliding_window_attention.ipynb`](./advanced_topics/02_sliding_window_attention.ipynb)

**What you'll learn**:
- Sliding window attention implementation
- Compare full vs windowed attention
- Measure effective context length (L × W)
- Benchmark speed and memory
- Visualize attention patterns

---

### Module 4.3: Pruning & Distillation (90 mins)
**Notebook**: [`advanced_topics/03_pruning_distillation.ipynb`](./advanced_topics/03_pruning_distillation.ipynb)

**What you'll learn**:
- Magnitude pruning implementation
- Structured pruning (heads, neurons)
- Knowledge distillation with temperature
- Distill Phi-3-Mini → SmolLM-1.7B
- Compare pruned vs distilled models

---

### Module 4.4: MCP Protocol (60 mins)
**Notebook**: [`advanced_topics/04_mcp_protocol.ipynb`](./advanced_topics/04_mcp_protocol.ipynb)

**What you'll learn**:
- Build MCP server (filesystem example)
- Create MCP client for model integration
- Implement tool calling with MCP
- Database access example
- API integration example

---

### Module 4.5: RLHF Pipeline (120 mins)
**Notebook**: [`advanced_topics/05_rlhf_pipeline.ipynb`](./advanced_topics/05_rlhf_pipeline.ipynb)

**What you'll learn**:
- Supervised fine-tuning stage
- Reward model training
- PPO implementation with trl
- DPO comparison
- Full RLHF pipeline end-to-end

---

### Module 4.6: Multimodal SLMs (90 mins)
**Notebook**: [`advanced_topics/06_multimodal_slms.ipynb`](./advanced_topics/06_multimodal_slms.ipynb)

**What you'll learn**:
- Load MiniCPM-V or similar
- Process image + text inputs
- Contrastive learning (CLIP-style)
- Fine-tune on custom image dataset
- Visualize image-text embeddings

---

### Module 4.7: Structured Output Generation (60 mins)
**Notebook**: [`advanced_topics/07_structured_output_generation.ipynb`](./advanced_topics/07_structured_output_generation.ipynb)

**What you'll learn**:
- JSON generation with outlines library
- Regex-constrained generation
- Grammar-based decoding
- Schema validation
- Compare constrained vs unconstrained

---

## 📚 **SECTION 5: MATHEMATICS & THEORY**

### Module 5.1: Linear Algebra Essentials (60 mins)
**Notebook**: [`mathematics/01_linear_algebra_essentials.ipynb`](./mathematics/01_linear_algebra_essentials.ipynb)

**What you'll learn**:
- ✅ Matrix operations for attention (Q, K, V)
- ✅ Dot product similarity visualization
- ✅ Softmax derivation and implementation
- ✅ Scaling by √d_k explanation
- ✅ Interactive attention heatmap builder

**Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmfast/slmhub/blob/main/notebooks/mathematics/01_linear_algebra_essentials.ipynb)

---

### Module 5.2: Backpropagation Deep Dive (75 mins)
**Notebook**: [`mathematics/02_backpropagation_deep_dive.ipynb`](./mathematics/02_backpropagation_deep_dive.ipynb)

**What you'll learn**:
- Chain rule visualization
- Manual gradient computation for attention
- Gradient flow analysis
- Vanishing/exploding gradient detection
- Residual connection gradient paths

---

### Module 5.3: Optimization Algorithms (90 mins)
**Notebook**: [`mathematics/03_optimization_algorithms.ipynb`](./mathematics/03_optimization_algorithms.ipynb)

**What you'll learn**:
- Adam optimizer from scratch
- AdamW implementation
- Lion optimizer comparison
- Learning rate schedules
- Benchmark optimizers on training task

---

### Module 5.4: Loss Functions (60 mins)
**Notebook**: [`mathematics/04_loss_functions.ipynb`](./mathematics/04_loss_functions.ipynb)

**What you'll learn**:
- Cross-entropy implementation
- Causal LM loss (shifted targets)
- Perplexity calculation
- Connection to maximum likelihood
- Visualize loss landscapes

---

### Module 5.5: Information Theory Basics (60 mins)
**Notebook**: [`mathematics/05_information_theory_basics.ipynb`](./mathematics/05_information_theory_basics.ipynb)

**What you'll learn**:
- Entropy calculation and visualization
- KL divergence implementation
- Temperature sampling effect on entropy
- DPO KL penalty visualization
- Distillation loss with KL

---

### Module 5.6: Probability & Sampling (75 mins)
**Notebook**: [`mathematics/06_probability_sampling.ipynb`](./mathematics/06_probability_sampling.ipynb)

**What you'll learn**:
- Greedy decoding implementation
- Temperature sampling with interactive demo
- Top-k sampling
- Top-p (nucleus) sampling
- Compare all strategies with visualizations

---

## 📚 **SECTION 6: MODELS HUB**

### Module 6.1: Model Database Schema (45 mins)
**Notebook**: [`models_hub/01_model_database_schema.ipynb`](./models_hub/01_model_database_schema.ipynb)

**What you'll learn**:
- YAML schema definition
- Model metadata structure
- Validation functions
- Example model entries
- Schema documentation generator

---

### Module 6.2: Interactive Model Comparison (60 mins)
**Notebook**: [`models_hub/02_interactive_model_comparison.ipynb`](./models_hub/02_interactive_model_comparison.ipynb)

**What you'll learn**:
- Load model database
- Filtering interface (size, license, benchmarks)
- Comparison table generation
- Radar chart visualization
- Export comparison results

---

### Module 6.3: Hardware Compatibility Matrix (45 mins)
**Notebook**: [`models_hub/03_hardware_compatibility_matrix.ipynb`](./models_hub/03_hardware_compatibility_matrix.ipynb)

**What you'll learn**:
- Memory calculation functions
- Speed estimation algorithms
- GPU compatibility checker
- Batch size calculator
- Interactive requirements tool

---

### Module 6.4: Model Leaderboards (45 mins)
**Notebook**: [`models_hub/04_model_leaderboards.ipynb`](./models_hub/04_model_leaderboards.ipynb)

**What you'll learn**:
- Leaderboard generation from database
- Category-based rankings
- Weighted scoring
- Filtering and sorting
- Live leaderboard updates

---

## 📚 **SECTION 7: COMMUNITY**

### Module 7.1: Discord Server (30 mins)
**Notebook**: [`community/01_discord_server.ipynb`](./community/01_discord_server.ipynb)

**What you'll learn**:
- Community structure overview
- Channel descriptions with code examples
- Bot integration examples
- Event scheduling code
- Community analytics

---

### Module 7.2: Contribution Guidelines (30 mins)
**Notebook**: [`community/02_contribution_guidelines.ipynb`](./community/02_contribution_guidelines.ipynb)

**What you'll learn**:
- PR template generator
- Model submission validator
- Tutorial checklist
- Code review guidelines
- Contribution tracking

---

### Module 7.3: Research Paper Summaries (45 mins)
**Notebook**: [`community/03_research_paper_summaries.ipynb`](./community/03_research_paper_summaries.ipynb)

**What you'll learn**:
- Paper parsing and summarization
- Implementation extraction
- Colab notebook generator from papers
- Paper comparison tool
- Citation network visualization

---

### Module 7.4: Industry Use Cases (45 mins)
**Notebook**: [`community/04_industry_use_cases.ipynb`](./community/04_industry_use_cases.ipynb)

**What you'll learn**:
- Use case template
- Deployment pattern examples
- Cost analysis calculator
- Performance benchmarking
- Case study generator

---

### Module 7.5: How to Contribute (30 mins)
**Notebook**: [`community/05_how_to_contribute.ipynb`](./community/05_how_to_contribute.ipynb)

**What you'll learn**:
- Contribution workflow diagram
- Model addition script
- Tutorial creation template
- Benchmark submission form
- Tool development guide

---

### Module 7.6: Code of Conduct (20 mins)
**Notebook**: [`community/06_code_of_conduct.ipynb`](./community/06_code_of_conduct.ipynb)

**What you'll learn**:
- CoC enforcement examples
- Community moderation tools
- Reporting mechanism
- Conflict resolution process
- Community health metrics

---

### Module 7.7: GitHub Discussions (30 mins)
**Notebook**: [`community/07_github_discussions.ipynb`](./community/07_github_discussions.ipynb)

**What you'll learn**:
- Discussion categorization
- Q&A bot examples
- Show-and-tell showcase
- Poll creation tool
- Discussion analytics

---

## 📚 **SECTION 9: ADVANCED ARCHITECTURES**

### Module 9.1: State Space Models (90 mins)
**Notebook**: [`advanced_architectures/01_state_space_models.ipynb`](./advanced_architectures/01_state_space_models.ipynb)

**What you'll learn**:
- ✅ SSM implementation from scratch
- ✅ Continuous vs discrete SSM
- ✅ O(n) complexity demonstration
- ✅ Long sequence handling
- ✅ Compare SSM vs Transformer

**Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmfast/slmhub/blob/main/notebooks/advanced_architectures/01_state_space_models.ipynb)

---

### Module 9.2: Mamba Architecture (120 mins)
**Notebook**: [`advanced_architectures/02_mamba_architecture.ipynb`](./advanced_architectures/02_mamba_architecture.ipynb)

**What you'll learn**:
- Selective SSM implementation
- Input-dependent A, B, C matrices
- Convolution for local context
- Hardware-aware algorithm
- Mamba block implementation

---

### Module 9.3: Mamba-2 & Mamba-3 Improvements (90 mins)
**Notebook**: [`advanced_architectures/03_mamba_2_3_improvements.ipynb`](./advanced_architectures/03_mamba_2_3_improvements.ipynb)

**What you'll learn**:
- SSD (State Space Duality) implementation
- Mamba-2 minimal code
- Mamba-3 improvements
- Performance comparisons
- Evolution visualization

---

### Module 9.4: Hybrid Architectures (90 mins)
**Notebook**: [`advanced_architectures/04_hybrid_architectures.ipynb`](./advanced_architectures/04_hybrid_architectures.ipynb)

**What you'll learn**:
- Jamba-style hybrid (Mamba + Attention)
- Layer placement strategies
- Attention frequency experiments
- Quality vs speed trade-offs
- Custom hybrid builder

---

### Module 9.5: RAG Advanced (90 mins)
**Notebook**: [`advanced_architectures/05_rag_advanced.ipynb`](./advanced_architectures/05_rag_advanced.ipynb)

**What you'll learn**:
- Multi-hop RAG implementation
- GraphRAG with NetworkX
- Self-RAG with retrieval decisions
- Comparison of RAG variants
- Long-context RAG

---

### Module 9.6: Speculative Decoding Deep Dive (90 mins)
**Notebook**: [`advanced_architectures/06_speculative_decoding_deep_dive.ipynb`](./advanced_architectures/06_speculative_decoding_deep_dive.ipynb)

**What you'll learn**:
- Draft + target model setup
- Parallel verification
- Acceptance/rejection logic
- Speedup measurement
- Optimal k selection

---

### Module 9.7: Quantization Theory (90 mins)
**Notebook**: [`advanced_architectures/07_quantization_theory.ipynb`](./advanced_architectures/07_quantization_theory.ipynb)

**What you'll learn**:
- Uniform quantization math
- Per-tensor vs per-channel
- GPTQ algorithm implementation
- Quantization error analysis
- Weight distribution visualization

---

## 📚 **SECTION 10: DEPLOYMENT & PRODUCTION**

### Module 10.1: Serving Infrastructure (90 mins)
**Notebook**: [`deployment/01_serving_infrastructure.ipynb`](./deployment/01_serving_infrastructure.ipynb)

**What you'll learn**:
- ✅ vLLM production setup with PagedAttention
- ✅ Async API server with FastAPI
- ✅ Batch inference optimization
- ✅ PagedAttention KV cache management
- ✅ Throughput benchmarking

**Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmfast/slmhub/blob/main/notebooks/deployment/01_serving_infrastructure.ipynb)

---

### Module 10.2: Monitoring & Observability (90 mins)
**Notebook**: [`deployment/02_monitoring_observability.ipynb`](./deployment/02_monitoring_observability.ipynb)

**What you'll learn**:
- Prometheus metrics integration
- Grafana dashboard setup
- Structured logging with structlog
- Latency tracking (p50, p90, p99)
- Error rate monitoring
- GPU utilization tracking

---

### Module 10.3: Cost Optimization (75 mins)
**Notebook**: [`deployment/03_cost_optimization.ipynb`](./deployment/03_cost_optimization.ipynb)

**What you'll learn**:
- Cost breakdown analysis
- Batch inference strategies
- Dynamic batching implementation
- Model quantization for cost reduction
- Response caching
- Auto-scaling strategies

---

## 📚 **SECTION 11: CUTTING-EDGE**

### Module 11.1: BitNet - 1.58-bit Quantization (120 mins)
**Notebook**: [`cutting_edge/01_bitnet_quantization.ipynb`](./cutting_edge/01_bitnet_quantization.ipynb)

**What you'll learn**:
- ✅ Ternary weight training (-1, 0, +1)
- ✅ BitLinear layer implementation
- ✅ Straight-through estimator
- ✅ Training from scratch with BitNet
- ✅ Performance comparison (memory, speed, quality)
- ✅ Energy efficiency analysis

**Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmfast/slmhub/blob/main/notebooks/cutting_edge/01_bitnet_quantization.ipynb)

---

### Module 11.2: Constitutional AI & Safety (90 mins)
**Notebook**: [`cutting_edge/02_constitutional_ai_safety.ipynb`](./cutting_edge/02_constitutional_ai_safety.ipynb)

**What you'll learn**:
- Constitutional AI workflow
- Critique and revision pipeline
- Safety filter implementation
- Toxicity detection
- PII detection
- Prompt injection prevention

---

### Module 11.3: Test-Time Compute Scaling (90 mins)
**Notebook**: [`cutting_edge/03_test_time_compute_scaling.ipynb`](./cutting_edge/03_test_time_compute_scaling.ipynb)

**What you'll learn**:
- Chain-of-thought prompting
- Self-consistency (majority voting)
- Best-of-N sampling
- Interactive reasoning visualization
- Quality vs compute trade-offs

---

### Module 11.4: Long Context Techniques (90 mins)
**Notebook**: [`cutting_edge/04_long_context_techniques.ipynb`](./cutting_edge/04_long_context_techniques.ipynb)

**What you'll learn**:
- YaRN (RoPE extension) implementation
- Streaming inference with sliding window
- Sink token preservation
- Context length extension beyond training
- Memory-efficient long-context handling

---

### Module 11.5: Emergent Abilities & Scaling (75 mins)
**Notebook**: [`cutting_edge/05_emergent_abilities_scaling.ipynb`](./cutting_edge/05_emergent_abilities_scaling.ipynb)

**What you'll learn**:
- Measuring emergent abilities
- Chinchilla scaling law implementation
- Optimal compute allocation
- Model size vs training data trade-offs
- Emergence visualization across scales

---

### Module 11.6: Multimodal Understanding (90 mins)
**Notebook**: [`cutting_edge/06_multimodal_understanding.ipynb`](./cutting_edge/06_multimodal_understanding.ipynb)

**What you'll learn**:
- Vision-language model architecture
- CLIP encoder integration
- Cross-attention implementation
- Image-text training pipeline
- Edge deployment for VLM (MiniCPM-V, MobileVLM)

---

## 📚 **SECTION 12: PRACTICAL PROJECTS**

### Module 12.1: Build a Code Assistant (120 mins)
**Notebook**: [`projects/01_code_assistant.ipynb`](./projects/01_code_assistant.ipynb)

**What you'll build**:
- ✅ Prepare coding dataset (The Stack)
- ✅ Fine-tune SmolLM with LoRA
- ✅ Code completion API
- ✅ VS Code extension integration
- ✅ Evaluation on HumanEval

**Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmfast/slmhub/blob/main/notebooks/projects/01_code_assistant.ipynb)

---

### Module 12.2: Personal Knowledge Base (RAG) (120 mins)
**Notebook**: [`projects/02_personal_knowledge_base.ipynb`](./projects/02_personal_knowledge_base.ipynb)

**What you'll build**:
- Complete RAG pipeline
- Document loading and chunking
- Embedding generation
- FAISS vector store setup
- Multi-hop RAG implementation
- Query interface
- Performance optimization

---

### Module 12.3: Function-Calling Agent (120 mins)
**Notebook**: [`projects/03_function_calling_agent.ipynb`](./projects/03_function_calling_agent.ipynb)

**What you'll build**:
- Tool definition schema
- ReAct pattern implementation
- Multi-tool orchestration
- Error handling and retries
- Agent evaluation framework

---

## 📚 **SECTION 13: ABOUT & RESOURCES**

### Module 13.1: Resources & Quick Reference (30 mins)
**Notebook**: [`about/01_resources_reference.ipynb`](./about/01_resources_reference.ipynb)

**What you'll find**:
- Official documentation links
- Must-read papers list
- Community resources
- Model selection flowchart
- Optimization techniques summary
- Quick reference tables

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
5. Module 5.1: Linear Algebra Essentials
6. Module 5.6: Probability & Sampling

### Advanced Track (60+ hours)
1. All modules in Sections 1-3
2. Section 4: Advanced Topics (MoE, RLHF, Multimodal)
3. Section 5: Mathematics & Theory (complete)
4. Section 6: Models Hub (tools and databases)
5. Section 7: Community (contribution workflows)
6. Section 9: Advanced Architectures (Mamba, Hybrid, RAG Advanced)
7. Complete all exercises
8. Build custom projects

### Production Track (70+ hours)
1. Complete Advanced Track
2. Section 10: Deployment & Production (serving, monitoring, cost optimization)
3. Section 11: Cutting-Edge (BitNet, Constitutional AI, long context)
4. Section 12: Projects (code assistant, RAG, agents)
5. Deploy real-world applications
6. Monitor and optimize production systems

### Research Track (80+ hours)
1. Complete Production Track
2. Deep dive into Section 9 (Advanced Architectures)
3. Section 11: Cutting-Edge research implementations
4. Implement research papers from Section 7.3
5. Contribute new models/tutorials
6. Publish case studies

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


import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const NOTEBOOKS_DIR = path.join(ROOT, 'notebooks');
const DOCS_DIR = path.join(ROOT, 'src/content/docs');
const PUBLIC_IMG_DIR = path.join(ROOT, 'public/images/notebooks');

// Map notebook filenames to their target MDX location
const NOTEBOOK_MAPPING = {
    // Section 1: Foundations
    'foundations/01_neural_networks_basics.ipynb': 'docs/learn/foundations/01-neural-networks-basics.mdx',
    'foundations/02_transformer_architecture.ipynb': 'docs/learn/foundations/02-transformer-architecture.mdx',
    'foundations/03_feedforward_normalization.ipynb': 'docs/learn/foundations/03-feedforward-normalization.mdx',
    'foundations/04_complete_transformer_block.ipynb': 'docs/learn/foundations/04-complete-transformer-block.mdx',
    'foundations/05_tokenization.ipynb': 'docs/learn/foundations/05-tokenization.mdx',
    'foundations/06_kv_cache.ipynb': 'docs/learn/foundations/06-kv-cache.mdx',
    'foundations/07_hardware_gpu_basics.ipynb': 'docs/learn/foundations/07-hardware-gpu-basics.mdx',
    'foundations/08_quantization_methods.ipynb': 'docs/learn/foundations/08-quantization-methods.mdx',
    'foundations/09_training_optimizations.ipynb': 'docs/learn/foundations/09-training-optimizations.mdx',
    'foundations/10_scaling_laws.ipynb': 'docs/learn/foundations/10-scaling-laws.mdx',
    
    // Section 2: Models
    'models/01_model_zoo.ipynb': 'docs/learn/models/01-model-zoo.mdx',
    'models/02_benchmarking.ipynb': 'docs/learn/models/02-benchmarking.mdx',
    'models/03_domain_specific.ipynb': 'docs/learn/models/03-domain-specific.mdx',
    'models/04_hardware_requirements.ipynb': 'docs/learn/models/04-hardware-requirements.mdx',
    
    // Section 3: Hands-On
    'hands_on/01_first_slm_10min.ipynb': 'docs/learn/hands-on/01-first-slm-10min.mdx',
    'hands_on/02_fine_tuning_lora.ipynb': 'docs/learn/hands-on/02-fine-tuning-lora.mdx',
    'hands_on/03_dpo_alignment.ipynb': 'docs/learn/hands-on/03-dpo-alignment.mdx',
    'hands_on/04_function_calling_agents.ipynb': 'docs/learn/hands-on/04-function-calling-agents.mdx',
    'hands_on/05_rag_system.ipynb': 'docs/learn/hands-on/05-rag-system.mdx',
    'hands_on/06_inference_optimization.ipynb': 'docs/learn/hands-on/06-inference-optimization.mdx',
    'hands_on/07_deployment.ipynb': 'docs/learn/hands-on/07-deployment.mdx',
    'hands_on/08_prompt_engineering.ipynb': 'docs/learn/hands-on/08-prompt-engineering.mdx',
    'hands_on/09_evaluation_monitoring.ipynb': 'docs/learn/hands-on/09-evaluation-monitoring.mdx',
    'hands_on/10_dataset_engineering.ipynb': 'docs/learn/hands-on/10-dataset-engineering.mdx',
    
    // Section 4: Advanced Topics
    'advanced_topics/01_mixture_of_experts.ipynb': 'docs/learn/advanced-topics/01-mixture-of-experts.mdx',
    'advanced_topics/02_sliding_window_attention.ipynb': 'docs/learn/advanced-topics/02-sliding-window-attention.mdx',
    'advanced_topics/03_pruning_distillation.ipynb': 'docs/learn/advanced-topics/03-pruning-distillation.mdx',
    'advanced_topics/04_mcp_protocol.ipynb': 'docs/learn/advanced-topics/04-mcp-protocol.mdx',
    'advanced_topics/05_rlhf_pipeline.ipynb': 'docs/learn/advanced-topics/05-rlhf-pipeline.mdx',
    'advanced_topics/06_multimodal_slms.ipynb': 'docs/learn/advanced-topics/06-multimodal-slms.mdx',
    'advanced_topics/07_structured_output_generation.ipynb': 'docs/learn/advanced-topics/07-structured-output-generation.mdx',
    
    // Section 5: Mathematics
    'mathematics/01_linear_algebra_essentials.ipynb': 'docs/learn/mathematics/01-linear-algebra-essentials.mdx',
    'mathematics/02_backpropagation_deep_dive.ipynb': 'docs/learn/mathematics/02-backpropagation-deep-dive.mdx',
    'mathematics/03_optimization_algorithms.ipynb': 'docs/learn/mathematics/03-optimization-algorithms.mdx',
    'mathematics/04_loss_functions.ipynb': 'docs/learn/mathematics/04-loss-functions.mdx',
    'mathematics/05_information_theory_basics.ipynb': 'docs/learn/mathematics/05-information-theory-basics.mdx',
    'mathematics/06_probability_sampling.ipynb': 'docs/learn/mathematics/06-probability-sampling.mdx',
    
    // Section 6: Models Hub
    'models_hub/01_model_database_schema.ipynb': 'docs/learn/models-hub/01-model-database-schema.mdx',
    'models_hub/02_interactive_model_comparison.ipynb': 'docs/learn/models-hub/02-interactive-model-comparison.mdx',
    'models_hub/03_hardware_compatibility_matrix.ipynb': 'docs/learn/models-hub/03-hardware-compatibility-matrix.mdx',
    'models_hub/04_model_leaderboards.ipynb': 'docs/learn/models-hub/04-model-leaderboards.mdx',
    
    // Section 7: Community
    'community/01_discord_server.ipynb': 'docs/learn/community/01-discord-server.mdx',
    'community/02_contribution_guidelines.ipynb': 'docs/learn/community/02-contribution-guidelines.mdx',
    'community/03_research_paper_summaries.ipynb': 'docs/learn/community/03-research-paper-summaries.mdx',
    'community/04_industry_use_cases.ipynb': 'docs/learn/community/04-industry-use-cases.mdx',
    'community/05_how_to_contribute.ipynb': 'docs/learn/community/05-how-to-contribute.mdx',
    'community/06_code_of_conduct.ipynb': 'docs/learn/community/06-code-of-conduct.mdx',
    'community/07_github_discussions.ipynb': 'docs/learn/community/07-github-discussions.mdx',
    
    // Section 9: Advanced Architectures
    'advanced_architectures/01_state_space_models.ipynb': 'docs/learn/advanced-architectures/01-state-space-models.mdx',
    'advanced_architectures/02_mamba_architecture.ipynb': 'docs/learn/advanced-architectures/02-mamba-architecture.mdx',
    'advanced_architectures/03_mamba_2_3_improvements.ipynb': 'docs/learn/advanced-architectures/03-mamba-2-3-improvements.mdx',
    'advanced_architectures/04_hybrid_architectures.ipynb': 'docs/learn/advanced-architectures/04-hybrid-architectures.mdx',
    'advanced_architectures/05_rag_advanced.ipynb': 'docs/learn/advanced-architectures/05-rag-advanced.mdx',
    'advanced_architectures/06_speculative_decoding_deep_dive.ipynb': 'docs/learn/advanced-architectures/06-speculative-decoding-deep-dive.mdx',
    'advanced_architectures/07_quantization_theory.ipynb': 'docs/learn/advanced-architectures/07-quantization-theory.mdx',
    
    // Section 10: Deployment
    'deployment/01_serving_infrastructure.ipynb': 'docs/learn/deployment/01-serving-infrastructure.mdx',
    'deployment/02_monitoring_observability.ipynb': 'docs/learn/deployment/02-monitoring-observability.mdx',
    'deployment/03_cost_optimization.ipynb': 'docs/learn/deployment/03-cost-optimization.mdx',
    
    // Section 11: Cutting-Edge
    'cutting_edge/01_bitnet_quantization.ipynb': 'docs/learn/cutting-edge/01-bitnet-quantization.mdx',
    'cutting_edge/02_constitutional_ai_safety.ipynb': 'docs/learn/cutting-edge/02-constitutional-ai-safety.mdx',
    'cutting_edge/03_test_time_compute_scaling.ipynb': 'docs/learn/cutting-edge/03-test-time-compute-scaling.mdx',
    'cutting_edge/04_long_context_techniques.ipynb': 'docs/learn/cutting-edge/04-long-context-techniques.mdx',
    'cutting_edge/05_emergent_abilities_scaling.ipynb': 'docs/learn/cutting-edge/05-emergent-abilities-scaling.mdx',
    'cutting_edge/06_multimodal_understanding.ipynb': 'docs/learn/cutting-edge/06-multimodal-understanding.mdx',
    
    // Section 12: Projects
    'projects/01_code_assistant.ipynb': 'docs/learn/projects/01-code-assistant.mdx',
    'projects/02_personal_knowledge_base.ipynb': 'docs/learn/projects/02-personal-knowledge-base.mdx',
    'projects/03_function_calling_agent.ipynb': 'docs/learn/projects/03-function-calling-agent.mdx',
    
    // Section 13: About
    'about/01_resources_reference.ipynb': 'docs/learn/about/01-resources-reference.mdx',
    
    // Legacy/Other notebooks (keep for backward compatibility)
    'ollama_quickstart.ipynb': 'docs/deploy/quickstarts/ollama.mdx',
    'embeddings.ipynb': 'docs/learn/fundamentals/embeddings.mdx',
    'architecture.ipynb': 'docs/learn/fundamentals/architecture.mdx',
    'agents.ipynb': 'docs/learn/concepts/agents.mdx',
    'vllm_quickstart.ipynb': 'docs/deploy/quickstarts/vllm.mdx',
    'llama_cpp_quickstart.ipynb': 'docs/deploy/quickstarts/llama-cpp.mdx',
    'start_here.ipynb': 'docs/start-here.mdx',
    'slm_vs_llm.ipynb': 'docs/learn/fundamentals/slm-vs-llm.mdx',
    'genai_basics.ipynb': 'docs/learn/fundamentals/genai-basics.mdx',
    'what_is_ai.ipynb': 'docs/learn/fundamentals/what-is-ai.mdx',
    'resources.ipynb': 'docs/tools/resources.mdx'
};

function ensureDir(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

function processNotebook(notebookName, targetMdxPath) {
    // Handle subdirectory paths (e.g., foundations/01_*.ipynb)
    const noteBookPath = path.join(NOTEBOOKS_DIR, notebookName);
    const mdxFullPath = path.join(DOCS_DIR, targetMdxPath);

    if (!fs.existsSync(noteBookPath)) {
        console.warn(`Notebook not found: ${notebookName}`);
        return;
    }

    const raw = fs.readFileSync(noteBookPath, 'utf-8');
    const nb = JSON.parse(raw);

    let mdxContent = "";
    let frontmatter = "";
    let title = "Updating...";
    let description = "Description";

    // Image extraction setup
    // Handle subdirectory paths for image output
    const nbBaseName = path.basename(notebookName, '.ipynb');
    const nbDir = path.dirname(notebookName);
    const imgOutputDir = nbDir ? path.join(PUBLIC_IMG_DIR, nbDir, nbBaseName) : path.join(PUBLIC_IMG_DIR, nbBaseName);
    ensureDir(imgOutputDir);

    let imgCount = 0;

    // 1. Process Cells
    for (let i = 0; i < nb.cells.length; i++) {
        const cell = nb.cells[i];
        const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;

        // Special handling for the first cell: Extract Frontmatter or Title
        if (i === 0 && cell.cell_type === 'markdown') {
            // If it looks like frontmatter, use it
            if (source.trim().startsWith('---')) {
                frontmatter = source.trim() + "\n";
                // Try to regex extract title/desc for the Widget
                const tMatch = source.match(/title:\s*["']?(.*?)["']?$/m);
                if (tMatch) title = tMatch[1];
                const dMatch = source.match(/description:\s*["']?(.*?)["']?$/m);
                if (dMatch) description = dMatch[1];
                continue;
            }
            // If header, use as title
            const lines = source.split('\n');
            if (lines[0].startsWith('# ')) {
                title = lines[0].replace('# ', '').trim();
                // Assume rest is description
                if (lines.length > 1) description = lines.slice(1).join(' ').trim().substring(0, 150) + "...";
                // Create frontmatter
                frontmatter = `---\ntitle: "${title}"\ndescription: "${description.replace(/"/g, '\\"')}"\n---\n`;
                // We can include the rest of the cell content, but skip the first line title usually
                mdxContent += source.replace(/^# .*$/m, '') + "\n\n";
                continue;
            }
        }

        if (cell.cell_type === 'markdown') {
            mdxContent += source + "\n\n";
        } else if (cell.cell_type === 'code') {
            // Code block
            mdxContent += `\`\`\`python\n${source}\n\`\`\`\n\n`;

            // Outputs
            if (cell.outputs && cell.outputs.length > 0) {
                for (const output of cell.outputs) {
                    if (output.output_type === 'stream') {
                        const text = Array.isArray(output.text) ? output.text.join('') : output.text;
                        mdxContent += `\`\`\`text\n${text}\n\`\`\`\n\n`;
                    } else if (output.data) {
                        // Handle Images
                        if (output.data['image/png']) {
                            const buffer = Buffer.from(output.data['image/png'], 'base64');
                            const imgFileName = `output_${imgCount}.png`;
                            const imgPath = path.join(imgOutputDir, imgFileName);
                            fs.writeFileSync(imgPath, buffer);

                            // Link in MDX (handle subdirectory paths)
                            const imgRelPath = nbDir ? `${nbDir}/${nbBaseName}` : nbBaseName;
                            const publicLink = `/slmhub/images/notebooks/${imgRelPath}/${imgFileName}`;
                            mdxContent += `![Output](${publicLink})\n\n`;
                            imgCount++;
                        }
                        // Handle Text
                        else if (output.data['text/plain']) {
                            const text = Array.isArray(output.data['text/plain']) ? output.data['text/plain'].join('') : output.data['text/plain'];
                            // Ignore if it's just a variable representation often mostly noise in docs unless explicit print
                            // But user wanted "outputs". Let's wrap in a nice block.
                            mdxContent += `> ${text.replace(/\n/g, '\n> ')}\n\n`;
                        }
                    }
                }
            }
        }
    }

    // 2. Assemble Final MDX
    const mdxDir = path.dirname(mdxFullPath);
    ensureDir(mdxDir);  // Ensure MDX output directory exists
    const componentsDir = path.join(ROOT, 'src/components');
    let relativePathToComponents = path.relative(mdxDir, componentsDir);
    relativePathToComponents = relativePathToComponents.replace(/\\/g, '/');

    const widgetImport = `import NotebookWidget from '${relativePathToComponents}/NotebookWidget.astro';\n\n`;
    // Handle subdirectory paths in notebook path
    const notebookPathForWidget = notebookName.includes('/') ? `notebooks/${notebookName}` : `notebooks/${notebookName}`;
    const widgetComponent = `<NotebookWidget \n  notebookPath="${notebookPathForWidget}"\n  title="${title}"\n  description="${description}"\n/>\n\n`;

    let finalContent = frontmatter + widgetImport + widgetComponent + mdxContent;

    // 3. Write
    fs.writeFileSync(mdxFullPath, finalContent, 'utf-8');
    console.log(`✓ Synced ${notebookName} -> ${targetMdxPath}`);
}

console.log("Starting Notebook Sync...");
for (const [nb, mdx] of Object.entries(NOTEBOOK_MAPPING)) {
    processNotebook(nb, mdx);
}
console.log("Sync Complete.");

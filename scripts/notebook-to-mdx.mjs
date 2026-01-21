
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
    'ollama_quickstart.ipynb': 'docs/deploy/quickstarts/ollama.mdx',
    'tokenization.ipynb': 'docs/learn/fundamentals/tokenization.mdx',
    'embeddings.ipynb': 'docs/learn/fundamentals/embeddings.mdx',
    'fine_tuning.ipynb': 'docs/learn/fundamentals/fine-tuning.mdx',
    'quantization.ipynb': 'docs/learn/fundamentals/quantization.mdx',
    'rag.ipynb': 'docs/learn/concepts/rag.mdx',
    'function_calling.ipynb': 'docs/learn/concepts/function-calling.mdx',
    'prompting.ipynb': 'docs/learn/fundamentals/prompting-for-slms.mdx',
    'architecture.ipynb': 'docs/learn/fundamentals/architecture.mdx',
    'agents.ipynb': 'docs/learn/concepts/agents.mdx',
    'datasets.ipynb': 'docs/learn/concepts/datasets.mdx',
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
    const nbBaseName = path.basename(notebookName, '.ipynb');
    const imgOutputDir = path.join(PUBLIC_IMG_DIR, nbBaseName);
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

                            // Link in MDX
                            const publicLink = `/slmhub/images/notebooks/${nbBaseName}/${imgFileName}`;
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
    const componentsDir = path.join(ROOT, 'src/components');
    let relativePathToComponents = path.relative(mdxDir, componentsDir);
    relativePathToComponents = relativePathToComponents.replace(/\\/g, '/');

    const widgetImport = `import NotebookWidget from '${relativePathToComponents}/NotebookWidget.astro';\n\n`;
    const widgetComponent = `<NotebookWidget \n  notebookPath="notebooks/${notebookName}"\n  title="${title}"\n  description="${description}"\n/>\n\n`;

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

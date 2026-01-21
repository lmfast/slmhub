---
title: Contributing
description: "Contribute docs like a developer: clear scope, reproducible claims, and tight reviews."
---


## The Notebook-First Approach

SLM Hub follows a **Notebook-First** architecture. This means technical guides should be written as Jupyter Notebooks (`.ipynb`) in the `notebooks/` directory.

### Why Notebooks?
- **Executability**: Users can click "Open in Colab" and run your code immediately.
- **Verification**: If the notebook runs, the docs are correct.
- **Rich Media**: Plots, tables, and outputs are automatically captured.

### How to add a new guide
1. Create your notebook in `/notebooks/your_guide.ipynb`.
2. Add a mapping in `scripts/notebook-to-mdx.mjs`.
3. Run `npm run sync:notebooks` to generate the documentation.
4. Commit both the `.ipynb` and the generated `.mdx`.

## Ways to contribute

- **Improve Notebooks**: Fix bugs in code cells or add better visualizations.
- **Add Tools**: Help us build better model pickers and hardware calculators.
- **Expand Content**: Create new quickstarts for emerging SLM frameworks.
- **Fix Docs**: Even if it's not a notebook, we appreciate clarity and typo fixes.

## Repo Conventions

- **Scripts**: Automation lives in `/scripts`.
- **Generated Docs**: Files in `src/content/docs` with an `import NotebookWidget` are auto-generated. **Do not edit these manually.** Edit the corresponding notebook instead.
- **Images**: Notebook images are automatically extracted to `public/images/notebooks/`.

## Pull request checklist

- [ ] Notebook runs end-to-end without errors.
- [ ] Code is clear and follows 2026 best practices (e.g., using `transformers` auto-loaders).
- [ ] MDX is correctly generated and referenced in the sidebar.
- [ ] Interactive buttons (Colab/Download) work.




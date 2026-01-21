# SLM Hub (Docs)

Developer-centric documentation for Small Language Models (SLMs): learn → choose a model → deploy → optimize.

## Documentation Structure

- **Home**: `src/pages/index.astro` (modern homepage)
- **Navigation**: `astro.config.mjs` (sidebar configuration)
- **Docs pages**: `src/content/docs/` (all markdown content)
- **Built with**: [Starlight](https://starlight.astro.build/) (Astro-based documentation framework)

This documentation is built with [Starlight](https://starlight.astro.build/) and [Astro](https://astro.build/), deployed automatically to GitHub Pages on every push to `main`.

## What this documentation is (and isn't)

- **Is**: practical guides, minimal explanations, copy/paste code that runs, and "how to choose" decision frameworks.
- **Isn't**: a benchmark leaderboard. We focus on **capability discovery** and **fit-for-purpose guidance**, not competitive scoreboards.

## Quickstart (local)

### Prerequisites

- Node.js 20+ and npm
- Python 3.11+ (for model sync scripts)

### Preview the documentation locally

```bash
# Install dependencies
npm install

# Migrate content (first time only, or after adding new files)
python scripts/migrate_to_starlight.py

# Start the local development server
npm run dev
```

Visit `http://localhost:4321` to see the documentation.

### Regenerate model metadata + pages

```bash
python scripts/sync_models.py
# Then migrate again to update Starlight structure
python scripts/migrate_to_starlight.py
```

### Build the site

```bash
npm run build
```

The built site will be in the `dist/` directory.

### Preview the built site

```bash
npm run preview
```

## Deployment

The documentation is automatically deployed to GitHub Pages on every push to the `main` branch via GitHub Actions. No manual steps required!

**To enable GitHub Pages:**
1. Go to your repository Settings → Pages
2. Under "Source", select "GitHub Actions"
3. The workflow (`.github/workflows/deploy.yml`) will handle the rest

**Important:** Update the following in `astro.config.mjs`:
- `site` URL (line 7) - Your GitHub Pages URL
- `base` path (line 8) - Your repository name
- `social.github` (line 19) - Your GitHub repository URL
- `editLink.baseUrl` (line 223) - Your GitHub repository URL

## Features (2026 Modern Standards)

✨ **Modern Design**
- Beautiful, responsive UI with dark mode
- Smooth animations and transitions
- Accessible and keyboard-friendly

🔍 **Powerful Search**
- Full-text search with Pagefind
- Fast and accurate results
- Search suggestions

📱 **Mobile-First**
- Responsive design
- Touch-friendly navigation
- Optimized for all screen sizes

⚡ **Performance**
- Static site generation
- Fast page loads
- Optimized assets

🎨 **Customizable**
- Custom CSS themes
- Component overrides
- Flexible configuration

## Automation overview

- **Model pages** (`models/generated/…`) are **auto-generated** from the Hugging Face Hub metadata.
- **Link checks** and **docs hygiene** run in GitHub Actions.
- **Documentation deployment** to GitHub Pages happens automatically on push.
- **Content migration** from root structure to Starlight structure is automated.

See `community/update-policy.md` for how we keep content current.

## Project Structure

```
slmhub/
├── src/
│   ├── content/
│   │   └── docs/          # All documentation content (migrated from root)
│   ├── components/         # Custom Astro components
│   ├── pages/              # Special pages (homepage, etc.)
│   ├── styles/             # Custom CSS
│   └── assets/             # Images, logos, etc.
├── scripts/
│   ├── migrate_to_starlight.py  # Migration script
│   └── sync_models.py           # Model sync script
├── astro.config.mjs        # Astro + Starlight configuration
├── package.json            # Node.js dependencies
└── requirements.txt        # Python dependencies
```

## Migration from MkDocs

If you're migrating from MkDocs, see `MIGRATION_STARLIGHT.md` for detailed instructions.



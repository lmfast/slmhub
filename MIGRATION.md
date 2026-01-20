# Migration from GitBook to MkDocs + GitHub Pages

This guide explains the migration from GitBook to MkDocs with GitHub Pages deployment.

## What Changed

1. **Documentation Engine**: GitBook → MkDocs with Material theme
2. **Deployment**: GitBook hosting → GitHub Pages (free, open source)
3. **Navigation**: `SUMMARY.md` → `mkdocs.yml` (converted automatically)
4. **Auto-deployment**: Now happens automatically on every push to `main` branch

## Setup Steps

### 1. Update mkdocs.yml

Edit `mkdocs.yml` and replace these placeholders:
- `YOUR_USERNAME` → Your GitHub username or organization name
- Update `site_url` and `repo_url` accordingly

Example:
```yaml
site_url: https://gaurav.github.io/slmhub
repo_url: https://github.com/gaurav/slmhub
```

### 2. Enable GitHub Pages

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select **"GitHub Actions"** (not "Deploy from a branch")
4. Save the settings

### 3. Push to GitHub

Once you push these changes to the `main` branch:
- The GitHub Actions workflow (`.github/workflows/deploy.yml`) will automatically:
  - Build the MkDocs site
  - Deploy it to GitHub Pages
  - Make it available at `https://YOUR_USERNAME.github.io/slmhub`

### 4. (Optional) Test Locally First

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Preview locally (uses wrapper script to set up docs directory)
bash scripts/mkdocs_wrapper.sh serve

# Visit http://127.0.0.1:8000
```

**Note:** We use a wrapper script (`scripts/mkdocs_wrapper.sh`) that creates a temporary `docs` directory with symlinks to your actual files. This allows MkDocs to work while keeping all your files in the root directory (no structure changes needed!).

## What Stays the Same

✅ All your markdown files remain unchanged
✅ Folder structure stays the same
✅ All your existing workflows (link-check, sync-models) continue to work
✅ No changes needed to your content

## Benefits

- ✅ **100% Free**: No paid plans, no limits
- ✅ **Open Source**: Fully open source stack
- ✅ **Auto-deploy**: Updates automatically on every push
- ✅ **Fast**: GitHub Pages CDN is fast and reliable
- ✅ **Customizable**: Material theme is highly customizable
- **No vendor lock-in**: You own everything

## Need Help?

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)

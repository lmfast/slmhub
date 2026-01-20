# Quick Start Guide - Starlight Documentation

Get your Starlight documentation site up and running in minutes!

## Prerequisites

- Node.js 20+ ([download](https://nodejs.org/))
- Python 3.11+ (for model sync scripts)
- Git

## Installation

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies (optional, for model sync)
pip install -r requirements.txt
```

## Configuration

1. **Update `astro.config.mjs`** - Replace `YOUR_USERNAME` with your GitHub username:
   ```javascript
   site: 'https://YOUR_USERNAME.github.io',
   base: '/slmhub',  // Change to your repo name
   social: {
     github: 'https://github.com/YOUR_USERNAME/slmhub',
   },
   ```

2. **Migrate content** (if not already done):
   ```bash
   python scripts/migrate_to_starlight.py
   ```

## Development

```bash
# Start development server
npm run dev

# Open http://localhost:4321
```

## Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Deploy

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Migrate to Starlight"
   git push
   ```

2. **Enable GitHub Pages:**
   - Go to your repo → Settings → Pages
   - Source: Select "GitHub Actions"
   - Save

3. **Done!** Your site will be live at:
   `https://YOUR_USERNAME.github.io/slmhub`

## Common Commands

```bash
npm run dev          # Start dev server
npm run build        # Build for production
npm run preview      # Preview production build
npm run sync-models  # Sync model data (requires Python)
```

## Need Help?

- See `STARLIGHT_SETUP.md` for detailed setup
- See `MIGRATION_STARLIGHT.md` for migration details
- [Starlight Docs](https://starlight.astro.build/)

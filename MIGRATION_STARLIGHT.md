# Migration from MkDocs to Starlight

This guide explains the complete migration from MkDocs to Starlight (Astro-based documentation framework).

## What Changed

1. **Documentation Engine**: MkDocs → Starlight (Astro)
2. **Structure**: Root markdown files → `src/content/docs/` structure
3. **Configuration**: `mkdocs.yml` → `astro.config.mjs`
4. **Build System**: Python-based → Node.js-based
5. **Deployment**: Same GitHub Pages, but with Astro build

## Why Starlight?

- ✅ **Modern 2026 Standards**: Built on Astro, the fastest static site generator
- ✅ **Better Performance**: Faster builds and page loads
- ✅ **Rich Features**: Built-in search, dark mode, responsive design
- ✅ **Component-Based**: Easy to customize and extend
- ✅ **Type-Safe**: TypeScript support out of the box
- ✅ **Active Development**: Regularly updated with new features

## Migration Steps

### 1. Install Dependencies

```bash
npm install
```

This installs:
- Astro
- Starlight
- Tailwind CSS
- Other dependencies

### 2. Run Migration Script

The migration script automatically converts your markdown files to Starlight's structure:

```bash
python scripts/migrate_to_starlight.py
```

This script:
- Moves all markdown files to `src/content/docs/`
- Adds proper frontmatter to each file
- Preserves your file structure
- Converts README.md to index.mdx

### 3. Update Configuration

Edit `astro.config.mjs` and update:

```javascript
site: 'https://YOUR_USERNAME.github.io',  // Your GitHub Pages URL
base: '/slmhub',                          // Your repository name
social: {
  github: 'https://github.com/YOUR_USERNAME/slmhub',
},
editLink: {
  baseUrl: 'https://github.com/YOUR_USERNAME/slmhub/edit/main/',
},
```

### 4. Test Locally

```bash
npm run dev
```

Visit `http://localhost:4321` to preview your site.

### 5. Build and Deploy

```bash
npm run build
```

The built site will be in `dist/` directory.

### 6. Enable GitHub Pages

1. Go to your repository Settings → Pages
2. Under "Source", select "GitHub Actions"
3. The workflow (`.github/workflows/deploy.yml`) will automatically deploy on push

## File Structure Changes

### Before (MkDocs)
```
slmhub/
├── README.md
├── start-here.md
├── learn/
│   └── README.md
└── mkdocs.yml
```

### After (Starlight)
```
slmhub/
├── src/
│   ├── content/
│   │   └── docs/
│   │       ├── index.mdx (from README.md)
│   │       ├── start-here.md
│   │       └── learn/
│   │           └── index.mdx (from README.md)
├── src/pages/
│   └── index.astro (homepage)
└── astro.config.mjs
```

## Key Differences

### Frontmatter

**MkDocs:**
```yaml
---
title: My Page
description: Page description
---
```

**Starlight:**
```yaml
---
title: My Page
description: Page description
---
```

(Actually, they're very similar! The migration script handles this automatically.)

### Navigation

**MkDocs** (`mkdocs.yml`):
```yaml
nav:
  - Home: README.md
  - Learn: learn/README.md
```

**Starlight** (`astro.config.mjs`):
```javascript
sidebar: [
  { label: 'Home', link: '/' },
  { label: 'Learn', link: '/learn/' },
]
```

### Code Blocks

Both support code blocks with syntax highlighting. Starlight uses Expressive Code for enhanced code blocks.

### Admonitions

**MkDocs:**
```markdown
!!! note "Note"
    This is a note.
```

**Starlight:**
```markdown
:::note
This is a note.
:::
```

The migration script can convert these automatically.

## Customization

### Custom CSS

Edit `src/styles/custom.css` to customize the theme.

### Custom Components

Add custom Astro components in `src/components/` and reference them in `astro.config.mjs`.

### Theme Colors

Edit the CSS variables in `src/styles/custom.css`:

```css
:root {
  --sl-color-accent: #6366f1;
  --sl-color-accent-low: #312e81;
  --sl-color-accent-high: #a5b4fc;
}
```

## Troubleshooting

### Build Errors

If you get build errors:
1. Check that all files have proper frontmatter
2. Run the migration script again: `python scripts/migrate_to_starlight.py`
3. Check for broken links: `npm run build` will show warnings

### Missing Pages

If pages are missing:
1. Check that they're in `src/content/docs/`
2. Verify the navigation in `astro.config.mjs`
3. Run the migration script again

### Styling Issues

If styles look wrong:
1. Clear the build cache: `rm -rf dist .astro`
2. Rebuild: `npm run build`
3. Check `src/styles/custom.css` for conflicts

## Features Available in Starlight

- ✅ **Full-text search** with Pagefind
- ✅ **Dark mode** toggle
- ✅ **Responsive design** (mobile-first)
- ✅ **Table of contents** on every page
- ✅ **Edit links** to GitHub
- ✅ **Last updated** dates
- ✅ **Multi-language support** (ready for i18n)
- ✅ **Custom components**
- ✅ **Expressive code blocks**
- ✅ **Social links**
- ✅ **Breadcrumbs**

## Next Steps

1. ✅ Run migration script
2. ✅ Update configuration
3. ✅ Test locally
4. ✅ Push to GitHub
5. ✅ Enable GitHub Pages
6. 🎉 Your site is live!

## Need Help?

- [Starlight Documentation](https://starlight.astro.build/)
- [Astro Documentation](https://docs.astro.build/)
- [Starlight GitHub](https://github.com/withastro/starlight)

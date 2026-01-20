# SLM Hub Design Overhaul (2026 Vision)

## Core Aesthetic: "Luminous Logic"
- **Base**: Clean, airy white background (`#ffffff` / `#f8fafc`) for maximum readability.
- **Accents**: Deep Indigo (`#4338ca`) mixed with vibrant Violet (`#7c3aed`) and a touch of Cyan (`#06b6d4`) for "electric" energy.
- **Texture**: Subtle mesh gradients, glassmorphism on sticky elements (navbar, sidebar), and micro-interactions.
- **Typography**: `Inter` (sans-serif) for UI, `JetBrains Mono` or `Fira Code` for code.

## Layout & Components

### 1. Homepage (`src/pages/index.astro`)
- **Hero Section**:
    - Large, bold typography: "Small Models, Big Impact."
    - Interactive SVG background (e.g., a constellation of nodes connecting).
    - "Get Started" and "Explore Models" prominent call-to-action buttons with glow effects.
- **Feature Grid**:
    - "Bento box" style layout for key sections (Learn, Deploy, Model Picker).
    - Hover effects: cards lift up, subtle internal glow.
- **Stats Bar**: Live-ish counters (e.g., "120+ Models Indexed").

### 2. Navigation
- **Navbar**:
    - Sticky with high blur (`backdrop-filter: blur(12px)`).
    - Semi-transparent white background (`rgba(255,255,255,0.8)`).
    - Search bar: Expanded, pill-shaped, with a "⌘K" indicator.
- **Sidebar**:
    - Clean hierarchy, subtle border-right, collapsible sections.

### 3. Documentation Content
- **Type Scale**: Modern, liquid typography.
- **Code Blocks**:
    - Custom "Open in Colab" button floating top-right of compatible code blocks.
    - "Copy" button (standard but styled).
    - Syntax highlighting: "One Dark Pro" or similar high-contrast theme.
- **Callouts**: Modern, flat design with color-coded left borders (Note, Tip, Warning).

## Technical Implementation
- **Framework**: Astro Starlight (Standard).
- **Styling**: Tailwind CSS + Custom CSS variables.
- **Interactive Elements**: Plain CSS animations (no heavy JS libs if possible).
- **Open in Colab**: Custom Astro component `<ColabButton />` that takes a path or infers it.

## "2026" Nuances
- **Dark Mode**: Deep blue/slate base (`#0f172a`), not just pitch black. Neon accents pop more.
- **Performance**: Zero CLS (Cumulative Layout Shift), instant transitions (View Transitions API).
- **Accessibility**: High contrast ratios, focus rings visible but aesthetic.
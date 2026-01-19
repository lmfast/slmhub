# SLM Hub (Docs)

Developer-centric documentation for Small Language Models (SLMs): learn → choose a model → deploy → optimize.

## How this repo maps to GitBook

- **Home**: `README.md`
- **Navigation**: `SUMMARY.md` (sidebar)
- **Docs pages**: folders like `learn/`, `models/`, `deploy/`, `tools/`, `community/`

GitBook can sync directly from GitHub. This repo is designed to work with GitBook’s Git Sync without requiring a build step.

## What this documentation is (and isn’t)

- **Is**: practical guides, minimal explanations, copy/paste code that runs, and “how to choose” decision frameworks.
- **Isn’t**: a benchmark leaderboard. We focus on **capability discovery** and **fit-for-purpose guidance**, not competitive scoreboards.

## Quickstart (local)

You don’t need to build anything to preview Markdown, but you can run automation scripts locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Regenerate model metadata + pages
python scripts/sync_models.py
```

## Automation overview

- **Model pages** (`models/generated/…`) are **auto-generated** from the Hugging Face Hub metadata.
- **Link checks** and **docs hygiene** run in GitHub Actions.

See `community/update-policy.md` for how we keep content current.



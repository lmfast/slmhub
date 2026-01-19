---
description: How we keep documentation current (and what we automate).
---

# Update policy (staying current)

## What updates automatically

- Hugging Face model metadata and generated pages:
  - `data/models.json`
  - `models/generated/`

These are updated by `scripts/sync_models.py`.

## What updates manually

- Tutorials and deployment patterns.
- Concept pages (fundamentals).

These should change only when:

- runtimes break,
- best practices shift,
- a page becomes misleading or too verbose.

## Cadence

- **Daily**: model directory sync (automation)
- **Weekly**: link checks
- **As needed**: tutorial fixes for breaking changes

## Freshness rules

- Prefer pinning versions/revisions for production guidance.
- If a claim depends on external behavior (API shape, runtime flags), add a link to the upstream docs and a note about the date verified.



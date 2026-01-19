---
title: RAG-Enhanced SLM
description: Retrieval + SLM is the default production architecture for many teams.
---

# RAG-enhanced SLM

## What this buys you

- Freshness (docs can change without re-training).
- Groundedness (answers based on retrieved context).
- Smaller model requirement (retrieval does the “knowledge” work).

## Hard parts

- Ingestion quality and chunking.
- Retrieval relevance.
- Output validation.

## Best practices

- Use citations in outputs.
- Add JSON schemas for machine-consumption.
- Add a “refuse if missing context” rule.



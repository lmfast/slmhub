---
title: Fine-Tuning Basics
description: Fine-tune small models with PEFT without turning your repo into a research project.
---

# Fine-tuning basics (PEFT)

## When fine-tuning is worth it

Fine-tune when:

- your task has consistent structure,
- prompting alone can’t close the gap,
- you can assemble a small, high-quality dataset (even a few thousand examples).

Prefer RAG when:

- you mainly need “fresh knowledge” retrieval,
- your outputs must reflect changing source-of-truth documents.

## The practical default in 2026

- **LoRA / QLoRA** for parameter-efficient updates.
- Keep training short; validate often.

## A minimal workflow

- Define acceptance prompts (before training).
- Prepare data with strict formatting.
- Train LoRA.
- Merge or ship adapter.
- Re-run acceptance prompts + a small adversarial set.



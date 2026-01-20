---
title: Multi-expert system
description: "Compose multiple small specialist models rather than one monolith."
---


## The idea

Use a small orchestrator (or rules) to route to specialists:

- code model for code,
- math model for math,
- vision model for images,
- general model for the rest.

## Why it works

- Operational flexibility: swap a specialist without retraining everything.
- Better unit economics than scaling one giant model.



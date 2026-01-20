---
title: Security
description: "Minimal security controls for SLM deployments."
---


## Default controls

- Input validation (size limits, schema checks)
- Rate limiting
- Authentication for all endpoints
- Audit logging (privacy-aware)

## Prompt injection realities

If you use tools/RAG:

- treat retrieved content as untrusted,
- separate system instructions from retrieved context,
- add allow-lists for tool calls.



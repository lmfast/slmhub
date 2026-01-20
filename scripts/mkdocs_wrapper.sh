#!/bin/bash
# Wrapper script for MkDocs that sets up the docs directory
# Usage: ./scripts/mkdocs_wrapper.sh [mkdocs command] [args...]
# Example: ./scripts/mkdocs_wrapper.sh serve
# Example: ./scripts/mkdocs_wrapper.sh build --strict

set -e

# Remove old docs directory if it exists
rm -rf docs

# Create docs directory
mkdir -p docs

# Create symlinks for all markdown files and directories we need
# Exclude build artifacts and non-documentation files
find . -maxdepth 1 -name "*.md" ! -name "MIGRATION.md" -exec ln -s ../{} docs/ \;
ln -s ../learn docs/ 2>/dev/null || true
ln -s ../models docs/ 2>/dev/null || true
ln -s ../deploy docs/ 2>/dev/null || true
ln -s ../tools docs/ 2>/dev/null || true
ln -s ../community docs/ 2>/dev/null || true

# Run mkdocs with the provided arguments
mkdocs "$@"

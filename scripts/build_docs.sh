#!/bin/bash
# Build script for MkDocs that preserves file structure
# Creates a temporary docs directory with symlinks to actual content

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

# Build the site
mkdocs build "$@"

# Clean up (optional - comment out if you want to inspect)
# rm -rf docs

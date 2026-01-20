#!/usr/bin/env python3
"""
Migration script to convert markdown files from root structure to Starlight structure.
Starlight expects files in src/content/docs/ with proper frontmatter.
"""

import os
import re
import shutil
from pathlib import Path

# Directories to migrate
MIGRATE_DIRS = [
    'learn',
    'models',
    'deploy',
    'tools',
    'community',
]

# Root markdown files to migrate
ROOT_FILES = [
    'start-here.md',
    'principles.md',
]

# Files to skip
SKIP_FILES = [
    'MIGRATION.md',
    'MIGRATION_STARLIGHT.md',
    'STARLIGHT_SETUP.md',
    'SUMMARY.md',
    'idea.md',
    'technical.md',
]

# Output directory
OUTPUT_DIR = Path('src/content/docs')

def extract_frontmatter(content):
    """Extract frontmatter from markdown content."""
    frontmatter_pattern = r'^---\n(.*?)\n---\n(.*)$'
    match = re.match(frontmatter_pattern, content, re.DOTALL)
    
    if match:
        frontmatter = match.group(1)
        body = match.group(2)
        return frontmatter, body
    return None, content

def add_starlight_frontmatter(content, file_path):
    """Add or update frontmatter for Starlight."""
    frontmatter, body = extract_frontmatter(content)
    
    # Extract title from first heading if not in frontmatter
    title_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else Path(file_path).stem.replace('-', ' ').title()
    
    # Extract description from frontmatter or first paragraph
    description = None
    if frontmatter:
        desc_match = re.search(r'description:\s*(.+)', frontmatter)
        if desc_match:
            description = desc_match.group(1).strip().strip('"\'')
    
    if not description:
        # Try to get first paragraph
        para_match = re.search(r'^([A-Z][^.!?]*[.!?])', body, re.MULTILINE)
        if para_match:
            description = para_match.group(1).strip()
    
    # Build new frontmatter
    new_frontmatter = f'---\ntitle: {title}\n'
    if description:
        # Escape double quotes
        description = description.replace('"', '\\"')
        new_frontmatter += f'description: "{description}"\n'
    new_frontmatter += '---\n\n'
    
    # Remove old frontmatter and first heading if it matches title
    if frontmatter:
        body = body  # Already extracted
    else:
        # Remove first heading if it's the same as title
        first_heading = re.match(r'^#\s+(.+)$', body, re.MULTILINE)
        if first_heading and first_heading.group(1).strip() == title:
            body = re.sub(r'^#\s+.+$\n', '', body, count=1, flags=re.MULTILINE)
    
    return new_frontmatter + body

def migrate_file(source_path, dest_path):
    """Migrate a single file."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add Starlight frontmatter
    new_content = add_starlight_frontmatter(content, source_path)
    
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f'Migrated: {source_path} -> {dest_path}')

def migrate_directory(source_dir, dest_dir):
    """Migrate a directory recursively."""
    source_path = Path(source_dir)
    dest_path = OUTPUT_DIR / dest_dir
    
    if not source_path.exists():
        print(f'Warning: {source_path} does not exist')
        return
    
    for file_path in source_path.rglob('*.md'):
        # Skip certain files
        if file_path.name in SKIP_FILES:
            continue
        
        # Calculate relative path
        rel_path = file_path.relative_to(source_path)
        dest_file = dest_path / rel_path
        
        # Convert README.md to index.mdx in subdirectories
        if file_path.name == 'README.md' and file_path.parent != source_path:
            dest_file = dest_file.parent / 'index.mdx'
        
        migrate_file(file_path, dest_file)

def main():
    """Main migration function."""
    print('Starting migration to Starlight structure...')
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Migrate root files
    for file_name in ROOT_FILES:
        source = Path(file_name)
        if source.exists():
            dest = OUTPUT_DIR / file_name
            migrate_file(source, dest)
    
    # Migrate directories
    for dir_name in MIGRATE_DIRS:
        migrate_directory(dir_name, dir_name)
    
    # Create index.md from README.md
    readme_path = Path('README.md')
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert to index with frontmatter
        index_content = add_starlight_frontmatter(content, 'README.md')
        index_path = OUTPUT_DIR / 'index.mdx'
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        print(f'Created index: {index_path}')
    
    print('\nMigration complete!')
    print(f'Files migrated to: {OUTPUT_DIR}')
    print('\nNext steps:')
    print('1. Review the migrated files in src/content/docs/')
    print('2. Run: npm install')
    print('3. Run: npm run dev')
    print('4. Update astro.config.mjs with your GitHub username')

if __name__ == '__main__':
    main()

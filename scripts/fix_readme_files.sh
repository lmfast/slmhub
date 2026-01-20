#!/bin/bash
# Convert README.md files to index.mdx in subdirectories

find src/content/docs -type d | while read dir; do
    readme_file="$dir/README.md"
    if [ -f "$readme_file" ]; then
        # Extract frontmatter and content
        if grep -q "^---" "$readme_file"; then
            # Has frontmatter, just rename
            mv "$readme_file" "$dir/index.mdx"
            echo "Renamed: $readme_file -> $dir/index.mdx"
        else
            # No frontmatter, add it
            title=$(basename "$dir" | sed 's/-/ /g' | awk '{for(i=1;i<=NF;i++)sub(/./,toupper(substr($i,1,1)),$i)}1')
            {
                echo "---"
                echo "title: $title"
                echo "---"
                echo ""
                cat "$readme_file"
            } > "$dir/index.mdx"
            rm "$readme_file"
            echo "Converted: $readme_file -> $dir/index.mdx"
        fi
    fi
done

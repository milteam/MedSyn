#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Running datasets for IFT generation..."

for file in "$SCRIPT_DIR"/*.py; do
    if [[ -f $file && "$file" != "$SCRIPT_DIR/utils.py" && "$file" != "$SCRIPT_DIR/merge_data.py" ]]; then
        echo "Running $file..."
        python3 "$file"
    fi
done

#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Running datasets for IFT generation..."

for file in "$SCRIPT_DIR"/*.py; do
    if [[ -f $file ]]; then
        echo "Running $file..."
        python3 "$file"
    fi
done

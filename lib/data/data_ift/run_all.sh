#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

MED_DATA_PERCENT=100
FILTRATE=False

echo "Running datasets for IFT generation..."

echo "Running datasets in medical domain..."
for file in "$SCRIPT_DIR"/med/*.py; do
    echo "Running $file..."
    python3 "$file"
done

if [ $MED_DATA_PERCENT -lt 100 ]; then
    echo "Running datasets in non-medical domain..."
    for file in "$SCRIPT_DIR"/non_med/*.py; do
        echo "Running $file..."
        python3 "$file"
    done
fi

if [ $FILTRATE = True ]; then
    echo "Data filtration..."
    python3 "$SCRIPT_DIR"/utils/filter_datasets.py
fi

echo "Merging data..."
python3 "$SCRIPT_DIR"/utils/merge_med.py --med_data_percent=$MED_DATA_PERCENT

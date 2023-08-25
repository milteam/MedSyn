# Filtering

```sh
pip install -q -r ./scripts/filtering/requirements.txt

./scripts/filtering/filter.py \
    --input-folder 'folder-template-containing-json-files-with-synthetic-records/results-*/*' \
    --output-path ./filtered.csv
```
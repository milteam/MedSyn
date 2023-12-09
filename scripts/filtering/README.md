# Filtering

## Creating a Docker Image for filtering

```sh
docker build -f Dockerfile.filtering -t medtext/filtering .
```

## Filtering model responses

To filter a JSONL file with model responses for default thresholds, adapt this command:

```sh
docker run --rm -it --gpus device=7 -v $MEDTEXT:/data \
    -e input=/data/model-responses/ru_llama2_7b_taskgen_ckpt_last_seed_42.jsonl \
    -e output=/data/filtered.csv \
    -e taskgen=/data/sampled_taskgen.csv \
    medtext/filtering
```

To specify custom thresholds for anamnesis precision and symptoms recall,
set `apt` and `srt` environment variables, respectively:
```sh
docker run --rm -it --gpus device=7 -v $MEDTEXT:/data \
    -e input=/data/model-responses/ru_llama2_7b_taskgen_ckpt_last_seed_42.jsonl \
    -e output=/data/filtered.csv \
    -e taskgen=/data/sampled_taskgen.csv \
    -e apt=0.8 \
    -e srt=0.77 \
    medtext/filtering
```
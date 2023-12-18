# Generation task

Filter information about diseases from WikiMed based on the following criteria:

* presence of clinical manifestations;
* ICD codes from WikiMed intersect with codes from RuMedTop3."

```sh
docker run --rm -v $MEDTEXT:/data medtext/taskgen python filter_wikimed.py \
    --wikimed-filepath /data/wikimed_diseases.csv \
    --rumedtop3-filepath /data/RuMedTop3/train_v1.jsonl \
    --output-filepath /data/rumedtop3_wikimed_manifestations.csv
```

Symptoms extraction with ChatGPT:
```sh
docker run --rm -v $MEDTEXT:/data medtext/taskgen python extract_symptoms.py \
    --input-filepath /data/rumedtop3_wikimed_manifestations.csv \
    --output-folder /data/.extracted_symptoms \
```

Create a generation task:

```sh
docker run -it --rm -v $MEDTEXT:/data medtext/taskgen python make_taskgen.py \
    --samples-filepath /data/samples.csv \
    --rumedtop3-filepath /data/RuMedTop3/train_v1.jsonl \
    --symptoms-filepath /data/rumedtop3_wikimed_symptoms_equipped_with_rumedprime.json \
    --output-folder /data
```

Generation with ChatGPT using task:

```sh
docker run -it --rm -v $MEDTEXT:/data medtext/taskgen python generate_anamneses.py \
    --taskgen-path $MEDTEXT/sampled_taskgen.csv \
    --output-folder /data/.gpt4-results
```
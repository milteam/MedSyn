# MedSyn: LLM-based Synthetic Medical Text Generation Framework

![](assets/pipeline.png)

The repository contains modules for generating clinical notes in Russian based on the target ICD-10 code. <br>
The proposed framework utilizes symptoms sampled from a medical knowledge graph and real clinical note examples for a target ICD code. <br>
Generating clinical notes without prior data is possible, but results may differ from those presented in the paper. <br>


## Data for instruction fine-tuning:
We provide all the collected data and data prepared for instruction fine-tuning. To access the data, download it from the [link](https://drive.google.com/drive/folders/1nElrx-pG2WXxdjZW_oYx4tHsWvUkarJy?usp=sharing):

* `datasets.zip` - collected datasets
* `data_ift.zip` - datasets processed in instruction fine-tuning format

Each sample in the instruction fine-tuning dataset is represented as: 
```json
   {
      "instruction": "Some kind of instruction.",
      "input": "Some prior information.",
      "output": "Desirable output."
   }
```

## Synthetic dataset:
Generated synthetic datasets containing 41,185 samples spanning 219 ICD-10 codes can be downloaded via the [link](https://drive.google.com/drive/folders/1nElrx-pG2WXxdjZW_oYx4tHsWvUkarJy?usp=sharing). <br>

`full-dataset-scored-anaon.csv`

| Data field | Description    |
| :---   | :--- |
| idx | Unique sample identifier. |
| ICD-10 | The targeted ICD-10 code used for prior data sampling. |
| generation_model | The model used for sample generation (GTP-3.5, GPT-4, LLaMA-7b, LLaMA-13b) |
| prompt | Prompt used for sample generation. |
| prior | Type of prior data used for sample generation. |
| example | Bool variable for the presence or absence of example during generation. |
| example | source Source of example (open-source RuMedPrime or private medical data). |
| response | Result of model generation. |
| symptoms | Symptoms used for prompt creation. |
| anamnesis | Clinical note example used as a style example in the prompt. |
| symptoms_recall | BERT-score for response and symptoms. |
| anamnesis_precision | BERT-score for response and anamnesis |


Part of real in-house clinical notes was hidden and replaced with `private_data` mark. <br>
Thirty samples from private real data were completely anonymized (by humans) and preserved in the dataset.


## Fine-tuned LLaMA-7b:
We provide a fine-tuned LLaMA-7b model checkpoint; it can be downloaded via the [link](https://drive.google.com/drive/folders/1nElrx-pG2WXxdjZW_oYx4tHsWvUkarJy?usp=sharing).

## Citation:

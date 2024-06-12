# MedSyn: LLM-based Synthetic Medical Text Generation Framework

![](assets/pipeline.png)

The repository contains modules for generating clinical notes in Russian based on the target ICD-10 code. <br>
The proposed framework utilizes symptoms sampled from a medical knowledge graph and real clinical note examples for a target ICD code. <br>
Generating clinical notes without prior data is possible, but results may differ from those presented in the paper. <br>


## Data:
We provide synthetic dataset and instruction fine-tuning dataset via HF datasets:
* [Synthetic dataset](https://huggingface.co/datasets/Glebkaa/MedSyn-synthetic)
* [IFT dataset](https://huggingface.co/datasets/Glebkaa/MedSyn-ift)


### Examples:
GPT-4 generation example:
![GPT-4 example](assets/gpt4_example.png)

LLaMA-7b generation example:
![LLaMa-7b](assets/llama_example.png)


## Fine-tuned LLaMA-7b:
We provide a fine-tuned LLaMA-7b model checkpoint; it can be downloaded via the [link](https://drive.google.com/drive/folders/1nElrx-pG2WXxdjZW_oYx4tHsWvUkarJy?usp=sharing).

## Citation:

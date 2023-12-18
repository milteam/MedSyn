

NLI huggingface
```bash
PYTHONPATH=./ CUDA_VISIBLE_DEVICES=5 python scripts/downstream/nli.py -t scripts/downstream/data/RuMedNLI/train_v1.jsonl -v scripts/downstream/data/RuMedNLI/dev_v1.jsonl -c nli_model -m hf -b DeepPavlov/rubert-base-cased -e 50
```

TOP# huggingface
```bash
PYTHONPATH=./ CUDA_VISIBLE_DEVICES=5 python scripts/downstream/top3.py -t scripts/downstream/data/RuMedTop3/train_v1.jsonl -v scripts/downstream/data/RuMedTop3/dev_v1.jsonl -c nli_model -m hf -b DeepPavlov/rubert-base-cased -e 50
```
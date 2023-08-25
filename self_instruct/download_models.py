import copy
import json
import os

from huggingface_hub import snapshot_download

model_dir = os.environ["MODEL_DIR"]
base_model = os.environ["BASE_MODEL"] 
lora_dir = os.environ["LORA_DIR"]
lora_weights = os.environ["LORA_WEIGHTS"]
token = os.environ["HF_TOKEN"]

# load base model
snapshot_download(repo_id=base_model, local_dir=model_dir,
                  ignore_patterns=["LICENSE", "README.md", "*.safetensors"], use_auth_token=token)
# load lora weights
snapshot_download(repo_id=lora_weights, local_dir=lora_dir,
                  ignore_patterns=["LICENSE", "README.md", "*.safetensors"], use_auth_token=token)

patch_base_model_name = False if "tloen" in lora_weights.lower() else True
patch_model_config = True if "decapoda" in base_model.lower() else False

if patch_base_model_name:
    replacement_keys = ["model_name", "base_model_name_or_path"]
    replacement_val = str(model_dir)
    
    configs_path = [f"{lora_dir}/training_config.json", f"{lora_dir}/adapter_config.json"]
    for cfg_filepath in configs_path:
        if os.path.exists(cfg_filepath):
            with open(cfg_filepath) as f:
                old_content = json.load(f)
            new_content = copy.deepcopy(old_content)
            
            for replacement_key in replacement_keys:
                if new_content.get(replacement_key):
                    print(f'{cfg_filepath}:')
                    print("Change model_name to a local checkpoint")
                    new_content[replacement_key] = replacement_val
                with open(cfg_filepath, "w") as f:
                    json.dump(new_content, f, indent=4)
        

if patch_model_config:
    replacements = {
        "tokenizer_config.json": {
            "tokenizer_class": "LlamaTokenizer",
            "model_max_length": 2048,
            "padding_side": "left",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "clean_up_tokenization_spaces": False,
            "special_tokens_map_file": "special_tokens_map.json",
        },
        "special_tokens_map.json": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "sep_token": "<s>",
            "unk_token": "<unk>",
        },
        "generation_config.json": {
            "_from_model_config": True,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2
        },
    }

    print('Patching model config...')
    for filename, new_content in replacements.items():
        print(f'{filename}:')
        with open(f"{model_dir}/{filename}") as fp:
            old_content = json.load(fp)
            print(f'    Original content: {old_content}')
            if old_content == new_content:
                print('    Already patched, skipping')
        print(f'    Updated content:  {new_content}')
        with open(f"{model_dir}/{filename}", "w") as fp:
            json.dump(new_content, fp, indent=4)

import yaml
import json
from prepare_codes import prepare_codes
from sampler import Sampler


def generate():
    with open("./scripts/sampler/cfg.yaml", "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    prepare_codes(cfg)
    sampler = Sampler(cfg)
    samples = sampler.generate_samples()

    with open(cfg["path_to_store"], "w") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    generate()

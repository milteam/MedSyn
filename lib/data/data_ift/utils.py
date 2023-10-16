import json
import os
from typing import List, Dict


def get_filepaths(root_dir: str) -> List[str]:
    filepaths = list()

    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            filepaths.append(os.path.join(path, name))

    return filepaths


def parse(root_dir: str) -> List[Dict]:
    filepaths = get_filepaths(root_dir)

    all_data = list()
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                all_data.append(row)

    return all_data

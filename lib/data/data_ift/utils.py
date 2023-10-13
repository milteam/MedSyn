import json
import os
from typing import List, Dict


def parse(root_dir: str) -> List[Dict]:
    filepaths = list()

    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            filepaths.append(os.path.join(path, name))

    all_data = list()
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                all_data.append(row)

    return all_data

from __future__ import annotations

import json
from typing import Dict


def get_hme_subsets(main_path: str) -> Dict[str, set[str]]:
    subsets = dict()
    for name in ['easy', 'medium', 'hard']:
        s = set()
        s.update(json.load(open(f'{main_path}/subset/{name}.json')))
        subsets[name] = s

    return subsets

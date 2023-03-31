from __future__ import annotations

from typing import Dict, Tuple
from zipfile import ZipFile
import json


def get_hme_subsets(main_path: str) -> Tuple[ZipFile, ZipFile, Dict[str, set[str]]]:
    subsets = dict()
    for name in ['easy', 'medium', 'hard']:
        s = set()
        s.update(json.load(open(f'{main_path}/subset/{name}.json')))
        subsets[name] = s

    return subsets

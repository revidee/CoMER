from typing import Dict, Tuple
from zipfile import ZipFile
import json

def get_hme_data(f: ZipFile) -> Tuple[ZipFile, ZipFile, Dict[str, set[str]]]:
    train_archive = ZipFile(f.open('train.zip'))
    test_archive = ZipFile(f.open('test.zip'))
    subsets = dict()
    for name in ['easy', 'medium', 'hard']:
        s = set()
        s.update(json.load(f.open(f'subset/{name}.json')))
        subsets[name] = s

    return train_archive, test_archive, subsets
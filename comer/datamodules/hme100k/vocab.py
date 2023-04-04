from __future__ import annotations
import os
from functools import lru_cache

from comer.datamodules.crohme.vocab import CROHMEVocab


@lru_cache()
def default_dict():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary.txt")


class HMEVocab(CROHMEVocab):
    def __init__(self, dict_path: str = default_dict()) -> None:
        super().__init__(dict_path)



def replace_textcircled_label(label: list[str]) -> list[str]:
    textcircled = '\\textcircled'
    text = '\\text'
    if textcircled in label:
        for i, token in enumerate(label):
            if token == textcircled:
                label[i] = text

    return label

vocab = HMEVocab()

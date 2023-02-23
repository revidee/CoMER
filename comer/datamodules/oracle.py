import math
from typing import List, Union, Hashable, Sequence, Dict

from comer.datamodules.crohme import DataEntry, vocab
from Levenshtein import distance


def _general_confidence(gt: Sequence[Hashable], pred: Sequence[Hashable]):
    wrong_pct_levenshtein = (distance(gt, pred, score_cutoff=(len(gt) - 1)) / len(gt))
    return math.log(1.0 - wrong_pct_levenshtein) if wrong_pct_levenshtein < 1.0 else float("-Inf")


def general_levenshtein(gt: Sequence[Hashable], pred: Sequence[Hashable]):
    return distance(gt, pred, score_cutoff=(len(gt) - 1))


class Oracle:
    def __init__(self, data: 'np.ndarray[Any, np.dtype[DataEntry]]'):
        self.label_dict: Dict[str, List[str]] = {}
        self.label_idx_dict: Dict[str, List[int]] = {}
        self.add_data(data)

    def confidence_indices(self, fname: str, pred: List[int]):
        return _general_confidence(self.label_idx_dict[fname], pred)

    def confidence_str(self, fname: str, pred: Union[List[str], str]):
        return _general_confidence(self.label_dict[fname], pred)

    def levenshtein_indices(self, fname: str, pred: List[int]) -> int:
        return general_levenshtein(self.label_idx_dict[fname], pred)

    def levenshtein_indices_nocutoff(self, fname: str, pred: List[int]) -> int:
        return distance(self.label_idx_dict[fname], pred)

    def levenshtein_str(self, fname: str, pred: Union[List[str], str]) -> int:
        return general_levenshtein(self.label_dict[fname], pred)

    def levenshtein_str_nocutoff(self, fname: str, pred: Union[List[str], str]) -> int:
        return distance(self.label_dict[fname], pred)

    def get_gt_indices(self, fname: str) -> List[int]:
        return self.label_idx_dict[fname]

    def add_data(self, data: 'np.ndarray[Any, np.dtype[DataEntry]]'):
        for entry in data:
            self.label_dict[entry.file_name] = entry.label
            self.label_idx_dict[entry.file_name] = vocab.words2indices(entry.label)
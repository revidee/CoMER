import math
from typing import List, Union, Hashable, Sequence

from comer.datamodules.crohme import DataEntry, vocab
from Levenshtein import distance


def _general_confidence(gt: Sequence[Hashable], pred: Sequence[Hashable]):
    return math.log(1.0 - (distance(gt, pred, score_cutoff=(len(gt) - 1)) / len(gt)))


class Oracle:
    def __init__(self, data: 'np.ndarray[Any, np.dtype[DataEntry]]'):
        self.label_dict = {}
        self.label_idx_dict = {}
        for entry in data:
            self.label_dict[entry.file_name] = entry.label
            self.label_idx_dict[entry.file_name] = vocab.words2indices(entry.label)

    def confidence_indices(self, fname: str, pred: List[int]):
        return _general_confidence(self.label_idx_dict[fname], pred)

    def confidence_str(self, fname: str, pred: Union[List[str], str]):
        return _general_confidence(self.label_dict[fname], pred)

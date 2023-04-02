from __future__ import annotations

from typing import Tuple, List
from zipfile import ZipFile

from comer.datamodules.crohme import BatchTuple, vocab
from comer.datamodules.crohme.batch import build_batch_split_from_entries
from comer.datamodules.ntcir.entry import extract_data_entries


def build_dataset(
        archive: ZipFile,
        folder: str,
        batch_size: int,
        used_vocab=vocab,
        unlabeled_pct: float = 0,
        sorting_mode: int = 0,  # 0 = nothing, 1 = random, 2 = sorted (asc), 3 = sorted (dsc)
) -> Tuple[List[BatchTuple], List[BatchTuple]]:
    return build_batch_split_from_entries(extract_data_entries(archive, folder, used_vocab=used_vocab), batch_size,
                                          unlabeled_pct=unlabeled_pct, sorting_mode=sorting_mode)
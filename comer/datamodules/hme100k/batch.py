from typing import Tuple, List, Optional, Dict
from zipfile import ZipFile

from comer.datamodules.crohme.batch import build_batch_split_from_entries, BatchTuple
from comer.datamodules.hme100k.entry import extract_data_entries


def build_dataset(
        archive: ZipFile,
        prefix: str,
        batch_size: int,
        limit: Optional[int] = None,
        subsets: Optional[Dict[str, set[str]]] = None,
        unlabeled_pct: float = 0,
        sorting_mode: int = 0,  # 0 = nothing, 1 = random, 2 = sorted (asc), 3 = sorted (dsc)
) -> Tuple[List[BatchTuple], List[BatchTuple]]:
    return build_batch_split_from_entries(
        extract_data_entries(archive, prefix, limit=limit, subsets=subsets),
        batch_size, unlabeled_pct=unlabeled_pct, sorting_mode=sorting_mode)
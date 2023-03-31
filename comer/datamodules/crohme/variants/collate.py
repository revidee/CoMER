from typing import List

from comer.datamodules.crohme import BatchTuple, Batch, create_batch_from_lists
from comer.datamodules.hme100k.vocab import vocab


# Used to transform a Lighting-Batch into some other form (here, our custom Batch)
def collate_fn(batch: List[BatchTuple]) -> Batch:
    assert len(batch) == 1
    return create_batch_from_lists(*(batch[0]))


def collate_fn_hme(batch: List[BatchTuple]) -> Batch:
    assert len(batch) == 1
    return create_batch_from_lists(*(batch[0]), used_vocab=vocab)

# Used to transform a Lighting-Batch into some other form (here, our custom Batch)
def collate_fn_remove_unlabeled(batch: List[BatchTuple]) -> Batch:
    assert len(batch) == 1
    return create_batch_from_lists(*(batch[0]), remove_unlabeled=True)
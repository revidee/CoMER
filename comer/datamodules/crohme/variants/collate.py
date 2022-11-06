from typing import List

from comer.datamodules.crohme import BatchTuple, Batch, create_batch_from_lists


# Used to transform a Lighting-Batch into some other form (here, our custom Batch)
def collate_fn(batch: List[BatchTuple]) -> Batch:
    assert len(batch) == 1
    return create_batch_from_lists(*(batch[0]))


# Used to transform a Lighting-Batch into some other form (here, our custom Batch)
def collate_fn_remove_unlabeled(batch: List[BatchTuple]) -> Batch:
    assert len(batch) == 1
    return create_batch_from_lists(*(batch[0]), remove_unlabeled=True)
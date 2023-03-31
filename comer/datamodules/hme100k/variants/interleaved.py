from typing import Optional, Any
from zipfile import ZipFile

import numpy as np

from comer.datamodules import CROHMEFixMatchInterleavedDatamodule
from comer.datamodules.crohme import get_splitted_indices, \
    build_batches_from_samples, DataEntry
from comer.datamodules.crohme.dataset import CROHMEDataset
from comer.datamodules.crohme.variants.collate import collate_fn_hme
from comer.datamodules.hme100k.batch import build_dataset
from comer.datamodules.hme100k.entry import extract_data_entries
from comer.datamodules.hme100k.extract import get_hme_data
from comer.datamodules.hme100k.vocab import vocab
from comer.datamodules.oracle import Oracle


class HMEInterleavedDatamodule(CROHMEFixMatchInterleavedDatamodule):

    def __init__(self, limit_val: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.limit_val = limit_val
        self.collate_fn = collate_fn_hme


    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            train, test, sets = get_hme_data(archive)
            if stage == "fit" or stage is None:
                assert 0.0 < self.unlabeled_pct < 1.0
                full_train_data: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(train, "train")

                labeled_indices, unlabeled_indices = get_splitted_indices(
                    full_train_data,
                    unlabeled_pct=self.unlabeled_pct,
                    sorting_mode=self.train_sorting
                )
                self.labeled_data, self.unlabeled_data = full_train_data[labeled_indices], full_train_data[
                    unlabeled_indices]

                # unlabeled train-split, used in the "pseudo-labeling" step
                # this uses the same batch size as the eval step, since inference requires more VRAM
                # (due to beam-search)
                self.pseudo_labeling_batches = build_batches_from_samples(
                    self.unlabeled_data,
                    self.eval_batch_size
                )

                self.unlabeled_factor = (1 / (1 - self.unlabeled_pct)) - 1

                # initialize the pseudo-labels with empty labels
                self.setup_pseudo_label_cache(self.unlabeled_data)

                # init oracle
                self.trainer.oracle = Oracle(self.unlabeled_data, used_vocab=vocab)

                self.val_dataset = CROHMEDataset(
                    build_dataset(test, 'test',  self.eval_batch_size, limit=self.limit_val, subsets=sets)[0],
                    "",
                    "",
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(test, 'test',  self.eval_batch_size)[0],
                    self.test_aug,
                    self.test_aug,
                )


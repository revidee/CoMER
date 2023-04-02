import logging
import os
from typing import  Optional
from zipfile import ZipFile

import numpy as np
import torch

from comer.datamodules import CROHMESupvervisedDatamodule
from comer.datamodules.crohme.batch import build_batch_split_from_entries, build_batches_from_samples
from comer.datamodules.crohme.dataset import CROHMEDataset
from comer.datamodules.crohme.variants.collate import collate_fn_hme, collate_fn

from comer.datamodules.crohme import vocab as vocabCROHME
from comer.datamodules.hme100k.vocab import vocab as vocabHME
from comer.datamodules.ntcir.batch import build_dataset
from comer.datamodules.ntcir.entry import extract_data_entries


class NTCIRSupervisedDatamodule(CROHMESupvervisedDatamodule):
    def __init__(self,
                 zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../../../ntcir.zip",
                 limit_val: int = 3000,
                 vocab: str = 'crohme',
                 **kwargs) -> None:
        super().__init__(zipfile_path=zipfile_path, **kwargs)

        assert vocab in ["crohme", "hme"]
        self.limit_val = limit_val
        self.used_vocab = vocabCROHME
        self.collate_fn = collate_fn
        if vocab == "hme":
            self.used_vocab = vocabHME
            self.collate_fn = collate_fn_hme

        logging.info(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = CROHMEDataset(
                    build_dataset(archive, "train", self.train_batch_size, used_vocab=self.used_vocab, unlabeled_pct=self.unlabeled_pct, sorting_mode=self.train_sorting)[0],
                    self.train_aug,
                    self.train_aug,
                )

                full_test = extract_data_entries(archive, "test", used_vocab=self.used_vocab)
                seed = torch.initial_seed()
                if seed > 2 ** 32 - 1:
                    seed = 7
                np.random.seed(seed)
                np.random.shuffle(full_test)

                self.val_dataset = CROHMEDataset(
                    build_batches_from_samples(full_test[:self.limit_val], self.eval_batch_size),
                    "",
                    "",
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, "test", self.eval_batch_size, used_vocab=self.used_vocab)[0],
                    self.test_aug,
                    self.test_aug,
                )
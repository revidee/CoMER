import logging
import os
from typing import  Optional
from zipfile import ZipFile

from comer.datamodules import CROHMESupvervisedDatamodule
from comer.datamodules.crohme.dataset import CROHMEDataset
from comer.datamodules.crohme.variants.collate import collate_fn_hme

from comer.datamodules.hme100k.batch import build_dataset
from comer.datamodules.hme100k.extract import get_hme_subsets


class HMESupvervisedDatamodule(CROHMESupvervisedDatamodule):
    def __init__(self, zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../../../hme100k.zip",
                 limit_val: int = 3000, **kwargs) -> None:
        super().__init__(zipfile_path=zipfile_path, **kwargs)
        self.limit_val = limit_val
        self.collate_fn = collate_fn_hme

        logging.info(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        sets = get_hme_subsets(self.zipfile_path)
        if stage == "fit" or stage is None:
            self.train_dataset = CROHMEDataset(
                build_dataset(self.zipfile_path, "train", self.train_batch_size, unlabeled_pct=self.unlabeled_pct, sorting_mode=self.train_sorting)[0],
                self.train_aug,
                self.train_aug,
            )
            self.val_dataset = CROHMEDataset(
                build_dataset(self.zipfile_path, 'test',  self.eval_batch_size, limit=self.limit_val, subsets=sets)[0],
                "",
                "",
            )
        if stage == "test" or stage is None:
            self.trainer.test_subsets = sets
            self.test_dataset = CROHMEDataset(
                build_dataset(self.zipfile_path, 'test',  self.eval_batch_size)[0],
                self.test_aug,
                self.test_aug,
            )
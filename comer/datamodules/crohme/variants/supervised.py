import os
from typing import  Optional
from zipfile import ZipFile

import pytorch_lightning as pl

from comer.datamodules.crohme.dataset import CROHMEDataset
from torch.utils.data.dataloader import DataLoader

from comer.datamodules.crohme import build_dataset
from comer.datamodules.crohme.variants.collate import collate_fn


class CROHMESupvervisedDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../../../data.zip",
            test_year: str = "2014",
            val_year: str = "2014",
            train_batch_size: int = 8,
            eval_batch_size: int = 4,
            num_workers: int = 5,
            train_aug: str = "weak",
            unlabeled_pct: float = 0.0,
            train_sorting: int = 1,
            test_aug: str = ""
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.zipfile_path = zipfile_path
        self.test_year = test_year
        self.val_year = val_year
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.train_aug = train_aug
        self.test_aug = test_aug
        self.unlabeled_pct = unlabeled_pct
        self.train_sorting = train_sorting

        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = CROHMEDataset(
                    build_dataset(archive, "train", self.train_batch_size, unlabeled_pct=self.unlabeled_pct, sorting_mode=self.train_sorting)[0],
                    self.train_aug,
                    self.train_aug,
                )
                self.val_dataset = CROHMEDataset(
                    build_dataset(archive, self.val_year, self.eval_batch_size)[0],
                    "",
                    "",
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size)[0],
                    self.test_aug,
                    self.test_aug,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

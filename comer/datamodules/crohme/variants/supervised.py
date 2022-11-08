import os
from typing import List, Optional
from zipfile import ZipFile

import pytorch_lightning as pl

from comer.datamodules.crohme.batch import create_batch_from_lists
from comer.datamodules.crohme.dataset import CROHMEDataset
from torch.utils.data.dataloader import DataLoader

from comer.datamodules.crohme import Batch, build_dataset, BatchTuple


# Used to transform a Lighting-Batch into some other form (here, our custom Batch)
def collate_fn(batch: List[BatchTuple]) -> Batch:
    assert len(batch) == 1
    return create_batch_from_lists(*(batch[0]))


class CROHMESupvervisedDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../../../data.zip",
            test_year: str = "2014",
            train_batch_size: int = 8,
            eval_batch_size: int = 4,
            num_workers: int = 5,
            scale_aug: bool = False,
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.zipfile_path = zipfile_path
        self.test_year = test_year
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug

        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = CROHMEDataset(
                    build_dataset(archive, "train", self.train_batch_size)[0],
                    True,
                    self.scale_aug,
                )
                self.val_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size)[0],
                    False,
                    self.scale_aug,
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size)[0],
                    False,
                    self.scale_aug,
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

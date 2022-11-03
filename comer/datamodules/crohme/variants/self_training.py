from typing import Optional
from zipfile import ZipFile

from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader

from comer.datamodules import CROHMESupvervisedDatamodule
from comer.datamodules.crohme import build_dataset
from comer.datamodules.crohme.dataset import CROHMEDataset
from comer.datamodules.crohme.variants.collate import collate_fn


class CROHMESelfTrainingDatamodule(CROHMESupvervisedDatamodule):
    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                train_labeled, train_unlabled = build_dataset(archive, "train", self.train_batch_size, unlabeled_factor=1)
                self.train_labeled_dataset = CROHMEDataset(
                    train_labeled,
                    True,
                    self.scale_aug,
                )
                self.train_unlabeled_dataset = CROHMEDataset(
                    train_unlabled,
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
            ConcatDataset([
                self.train_labeled_dataset, self.train_unlabeled_dataset
            ]),
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

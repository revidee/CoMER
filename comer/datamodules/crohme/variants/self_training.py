from typing import Optional
from zipfile import ZipFile

from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader

from comer.datamodules import CROHMESupvervisedDatamodule
from comer.datamodules.crohme import build_dataset
from comer.datamodules.crohme.dataset import CROHMEDataset
from comer.datamodules.crohme.variants.collate import collate_fn, collate_fn_remove_unlabeled


class CROHMESelfTrainingDatamodule(CROHMESupvervisedDatamodule):
    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                train_labeled, train_unlabled = build_dataset(archive, "train", self.train_batch_size,
                                                              unlabeled_factor=0.5)

                # "dynamic dataset"
                self.train_unlabled_ds = train_unlabled
                self.trainer.unlabeled_pseudo_labels = [[[] for _ in unl_batch[0]] for unl_batch in self.train_unlabled_ds]

                # "static" datasets
                self.train_labeled_dataset = CROHMEDataset(
                    train_labeled,
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

    def filter_unlabeled(self):
        if self.trainer is None or self.trainer.unlabeled_pseudo_labels is None:
            return
        filtered_batches = []
        for idx, pseudo_labels_for_batch in enumerate(self.trainer.unlabeled_pseudo_labels):
            for single_item_label in pseudo_labels_for_batch:
                if len(single_item_label) > 0:
                    src_batch = self.train_unlabled_ds[idx]
                    src_batch[2].clear()
                    src_batch[2].extend(pseudo_labels_for_batch)
                    filtered_batches.append(src_batch)
                    break

        return filtered_batches

    def all_unlabeled_with_potential_pseudos(self):
        if self.trainer is None or self.trainer.unlabeled_pseudo_labels is None:
            return self.train_unlabled_ds
        for idx, src_batch in enumerate(self.train_unlabled_ds):
            pseudos = self.trainer.unlabeled_pseudo_labels[idx]
            self.train_unlabled_ds[idx][2].clear()
            self.train_unlabled_ds[idx][2].extend(pseudos)
        return self.train_unlabled_ds

    def train_dataloader(self):
        train_sets = [
            self.train_labeled_dataset
        ]
        filtered_data = self.filter_unlabeled()
        if len(filtered_data) > 0:
            train_sets.append(CROHMEDataset(
                filtered_data,
                True,
                self.scale_aug,
            ))
        return DataLoader(
            ConcatDataset(train_sets),
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn_remove_unlabeled,
        )

    def val_dataloader(self):
        return [DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        ), DataLoader(
            CROHMEDataset(
                self.all_unlabeled_with_potential_pseudos(),
                False,
                False,
            ),
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )]

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
                labeled_ds, self.train_unlabeled_ds = build_dataset(
                    archive,
                    "train",
                    self.train_batch_size,
                    unlabeled_pct=self.unlabeled_pct,
                    sorting_mode=self.train_sorting
                )
                self.trainer.unlabeled_pseudo_labels = [[[] for _ in unl_batch[0]] for unl_batch in self.train_unlabeled_ds]

                # "static" datasets
                self.train_labeled_dataset = CROHMEDataset(
                    labeled_ds,
                    "weak",
                )
                self.val_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size)[0],
                    "",
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size)[0],
                    "",
                )

    def filter_unlabeled(self):
        if self.trainer is None or self.trainer.unlabeled_pseudo_labels is None:
            return
        filtered_batches = []
        for idx, pseudo_labels_for_batch in enumerate(self.trainer.unlabeled_pseudo_labels):
            for single_item_label in pseudo_labels_for_batch:
                if len(single_item_label) > 0:
                    src_batch = self.train_unlabeled_ds[idx]
                    src_batch[2].clear()
                    src_batch[2].extend(pseudo_labels_for_batch)
                    filtered_batches.append(src_batch)
                    break

        return filtered_batches

    def all_unlabeled_with_potential_pseudos(self):
        if self.trainer is None or self.trainer.unlabeled_pseudo_labels is None:
            return self.train_unlabeled_ds
        for idx, src_batch in enumerate(self.train_unlabeled_ds):
            pseudos = self.trainer.unlabeled_pseudo_labels[idx]
            self.train_unlabeled_ds[idx][2].clear()
            self.train_unlabeled_ds[idx][2].extend(pseudos)
        return self.train_unlabeled_ds

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_labeled_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

        filtered_data = self.filter_unlabeled()
        if len(filtered_data) > 0:
            return {
                "labeled": train_dl,
                "unlabeled": DataLoader(
                    CROHMEDataset(
                        filtered_data,
                        "strong"
                    ),
                    shuffle=True,
                    num_workers=self.num_workers,
                    collate_fn=collate_fn_remove_unlabeled,
                )
            }
        return {
            "labeled": train_dl
        }

    def val_dataloader(self):
        return [DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        ), DataLoader(
            CROHMEDataset(
                self.all_unlabeled_with_potential_pseudos(),
                "",
            ),
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )]

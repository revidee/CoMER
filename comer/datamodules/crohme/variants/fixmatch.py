from typing import Optional, Any
from zipfile import ZipFile

from torch.utils.data.dataloader import DataLoader

from comer.datamodules import CROHMESupvervisedDatamodule
from comer.datamodules.crohme import build_dataset, extract_data_entries, get_splitted_indices, \
    build_batches_from_samples, DataEntry
from comer.datamodules.crohme.dataset import CROHMEDataset
from comer.datamodules.crohme.variants.collate import collate_fn, collate_fn_remove_unlabeled


class CROHMESelfTrainingDatamodule(CROHMESupvervisedDatamodule):
    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                assert 0.0 < self.unlabeled_pct < 1.0
                full_train_data: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "train")

                labeled_indices, unlabeled_indices = get_splitted_indices(
                    full_train_data,
                    unlabeled_pct=self.unlabeled_pct,
                    sorting_mode=self.train_sorting
                )
                labeled_data, unlabeled_data = full_train_data[labeled_indices], full_train_data[unlabeled_indices]
                # labeled train-split
                self.train_labeled_dataset = CROHMEDataset(
                    build_batches_from_samples(
                        labeled_data,
                        self.train_batch_size,
                        is_labled=True,
                        batch_imagesize=int(10e10),
                        max_imagesize=int(10e10)
                    ),
                    "weak",
                )

                # unlabeled train-split, used in the "pseudo-labeling" step
                # this uses the same batch size as the eval step, since inference requires more VRAM
                self.pseudo_labeling_batches = build_batches_from_samples(
                    unlabeled_data,
                    self.eval_batch_size,
                    is_labled=True
                )

                # unlabeled train-split, used in the train-step. This uses a larger batch_size than the usual
                # train_batch_size, since the losses for both are averaged together
                # Like FixMatch, we increase the batch-size by the factor
                unlabeled_factor = (1 / (1 - self.unlabeled_pct)) - 1
                unlabeled_batch_size = self.train_batch_size * unlabeled_factor
                if unlabeled_batch_size < 1:
                    unlabeled_batch_size = 1

                self.pseudo_labeled_batches = build_batches_from_samples(
                    unlabeled_data,
                    unlabeled_batch_size,
                    batch_imagesize=int(10e10),
                    max_imagesize=int(10e10),
                    is_labled=False
                )

                # initialize the pseudo-labels with empty labels
                self.trainer.unlabeled_pseudo_labels = {}
                for entry in unlabeled_data:
                    self.trainer.unlabeled_pseudo_labels[entry.file_name] = []

                self.val_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size)[0],
                    "",
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size)[0],
                    "",
                )

    def add_labels_to_batches(self, batches: list[tuple[list[str], list[ndarray], list[list[str]], bool, int]]):
        if self.trainer is None or self.trainer.unlabeled_pseudo_labels is None:
            return batches
        for idx, src_batch in enumerate(batches):
            src_batch[2].clear()
            src_batch[2].extend(
                [self.trainer.unlabeled_pseudo_labels[fname] for fname in src_batch[0]]
            )
        return batches

    def get_unlabeled_for_train(self):
        return self.add_labels_to_batches(self.pseudo_labeled_batches)

    def get_unlabeled_for_pseudo_labeling(self):
        return self.add_labels_to_batches(self.pseudo_labeling_batches)

    def train_dataloader(self):
        return {
            "labeled": DataLoader(
                self.train_labeled_dataset,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
            ),
            "unlabeled": DataLoader(
                CROHMEDataset(
                    self.get_unlabeled_for_train(),
                    "strong"
                ),
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn_remove_unlabeled,
            )
        }

    def val_dataloader(self):
        return [DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        ), DataLoader(
            CROHMEDataset(
                self.get_unlabeled_for_pseudo_labeling(),
                "weak",
            ),
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )]

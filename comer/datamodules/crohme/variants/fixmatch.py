from typing import Optional, Any, List, Dict
from zipfile import ZipFile

from torch.utils.data.dataloader import DataLoader

from comer.datamodules import CROHMESupvervisedDatamodule
from comer.datamodules.crohme import build_dataset, extract_data_entries, get_splitted_indices, \
    build_batches_from_samples, DataEntry, BatchTuple
from comer.datamodules.crohme.batch import MaybePartialLabel
from comer.datamodules.crohme.dataset import CROHMEDataset
from comer.datamodules.crohme.variants.collate import collate_fn_remove_unlabeled


class CROHMEFixMatchDatamodule(CROHMESupvervisedDatamodule):

    def __init__(self, unlabeled_strong_aug: str = "strong", unlabeled_weak_aug: str = "weak", **kwargs):
        super().__init__(**kwargs)
        self.unlabeled_strong_augmentation = unlabeled_strong_aug
        self.unlabeled_weak_augmentation = unlabeled_weak_aug

    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                assert 0.0 <= self.unlabeled_pct <= 1.0
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
                        batch_imagesize=int(10e10),
                        max_imagesize=int(10e10)
                    ),
                    "weak",
                    "weak"
                )

                # unlabeled train-split, used in the "pseudo-labeling" step
                # this uses the same batch size as the eval step, since inference requires more VRAM
                # (due to beam-search)
                self.pseudo_labeling_batches = build_batches_from_samples(
                    unlabeled_data,
                    self.eval_batch_size,
                )

                # unlabeled train-split, used in the train-step. This uses a larger batch_size than the usual
                # train_batch_size, since the losses for both are averaged together
                # Like FixMatch, we increase the batch-size by the factor
                unlabeled_factor = (1 / (1 - self.unlabeled_pct)) - 1
                unlabeled_batch_size = int(self.train_batch_size * unlabeled_factor)
                if unlabeled_batch_size < 1:
                    unlabeled_batch_size = 1

                self.pseudo_labeled_batches = build_batches_from_samples(
                    unlabeled_data,
                    unlabeled_batch_size,
                    batch_imagesize=int(10e10),
                    max_imagesize=int(10e10),
                )

                # initialize the pseudo-labels with empty labels
                self.setup_pseudo_label_cache(unlabeled_data)

                self.val_dataset = CROHMEDataset(
                    build_dataset(archive, self.val_year, self.eval_batch_size)[0],
                    "",
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size)[0],
                    "",
                )

    def setup_pseudo_label_cache(self, unlabeled_data: List[DataEntry]):
        # initialize the pseudo-labels with empty labels
        self.trainer.unlabeled_pseudo_labels: Dict[str, MaybePartialLabel] = {}
        for entry in unlabeled_data:
            self.trainer.unlabeled_pseudo_labels[entry.file_name] = (False, [], None)

    def add_labels_to_batches(self, batches: List[BatchTuple]):
        if self.trainer is None or "unlabeled_pseudo_labels" not in self.trainer:
            return batches
        for idx, src_batch in enumerate(batches):
            src_batch[2].clear()
            src_batch[2].extend(
                [self.trainer.unlabeled_pseudo_labels[fname] for fname in src_batch[0]]
            )
        return batches

    def get_unlabeled_for_train(self):
        self.add_labels_to_batches(self.pseudo_labeled_batches)
        filtered_batches = []
        for idx, src_batch in enumerate(self.pseudo_labeled_batches):
            for (is_partial, label, label_r2l) in src_batch[2]:
                if (label is not None and len(label) > 0) or (label_r2l is not None and len(label_r2l) > 0):
                    filtered_batches.append(src_batch)
                    break
        return filtered_batches

    def get_unlabeled_for_pseudo_labeling(self):
        return self.add_labels_to_batches(self.pseudo_labeling_batches)

    def train_dataloader(self):
        unlabeled_with_pseudos = self.get_unlabeled_for_train()
        labeled = DataLoader(
            self.train_labeled_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        if len(unlabeled_with_pseudos) == 0:
            return {
                "labeled": labeled
            }
        return {
            "labeled": labeled,
            "unlabeled": DataLoader(
                CROHMEDataset(
                    unlabeled_with_pseudos,
                    self.unlabeled_strong_augmentation,
                    self.unlabeled_strong_augmentation
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
            collate_fn=self.collate_fn,
        ), DataLoader(
            CROHMEDataset(
                self.pseudo_labeling_batches,
                self.unlabeled_weak_augmentation,
                self.unlabeled_weak_augmentation,
            ),
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )]

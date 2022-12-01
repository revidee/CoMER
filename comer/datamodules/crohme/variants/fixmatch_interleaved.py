from typing import Optional, Any, List
from zipfile import ZipFile

from torch.utils.data.dataloader import DataLoader

from comer.datamodules.crohme import build_dataset, extract_data_entries, get_splitted_indices, \
    build_batches_from_samples, DataEntry, BatchTuple
from comer.datamodules.crohme.dataset import CROHMEDataset
from comer.datamodules.crohme.variants.collate import collate_fn, collate_fn_remove_unlabeled
from comer.datamodules.crohme.variants.fixmatch import CROHMEFixMatchDatamodule


class CROHMEFixMatchInterleavedDatamodule(CROHMEFixMatchDatamodule):
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

                self.pseudo_labeled_batches = build_batches_from_samples(
                    unlabeled_data,
                    self.train_batch_size,
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

    def train_dataloader(self):
        unlabeled_with_pseudos = self.get_unlabeled_for_train()
        if len(unlabeled_with_pseudos) == 0:
            return DataLoader(
                self.train_labeled_dataset,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
            )
        # Interleave unlabeled / labeled batches
        factor = (1 / (1 - self.unlabeled_pct)) - 1

        # TODO: interleave based on the factor


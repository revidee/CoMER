from typing import Optional, Any, List
from zipfile import ZipFile

import numpy as np
from torch.utils.data.dataloader import DataLoader

from comer.datamodules.crohme import build_dataset, extract_data_entries, get_splitted_indices, \
    build_batches_from_samples, DataEntry, build_interleaved_batches_from_samples
from comer.datamodules.crohme.dataset import CROHMEDataset
from comer.datamodules.crohme.variants.collate import collate_fn
from comer.datamodules.crohme.variants.fixmatch import CROHMEFixMatchDatamodule
from comer.datamodules.oracle import Oracle


class CROHMEFixMatchInterleavedTestAsUnlabeledDatamodule(CROHMEFixMatchDatamodule):
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
                self.labeled_data, self.unlabeled_data = full_train_data[labeled_indices], full_train_data[
                    unlabeled_indices]

                for year in ["2014", "2016", "2019"]:
                    self.unlabeled_data = np.append(self.unlabeled_data, extract_data_entries(archive, year))

                # unlabeled train-split, used in the "pseudo-labeling" step
                # this uses the same batch size as the eval step, since inference requires more VRAM
                # (due to beam-search)
                self.pseudo_labeling_batches = build_batches_from_samples(
                    self.unlabeled_data,
                    self.eval_batch_size
                )

                self.unlabeled_factor = (1 / (1 - self.unlabeled_pct)) - 1

                # initialize the pseudo-labels with empty labels
                self.setup_pseudo_label_cache(self.unlabeled_data)

                # init oracle
                self.trainer.oracle = Oracle(self.unlabeled_data)

                self.val_dataset = CROHMEDataset(
                    build_dataset(archive, self.val_year, self.eval_batch_size)[0],
                    "",
                    "",
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size)[0],
                    "",
                    "",
                )
    def get_interleaved_train_batches(self):
        filtered_unlabeled_entries = []
        total_labeled = 0
        self.unlabeled_data: List[DataEntry]
        for data_entry in self.unlabeled_data:
            # add label from gpu-shared labeling cache
            data_entry.is_partial, data_entry.label, data_entry.label_r2l\
                = self.trainer.unlabeled_pseudo_labels[data_entry.file_name]
            # if it has a valid label, add it to the "filtered" data-set
            if (data_entry.label is not None and len(data_entry.label) > 0)\
                    or (data_entry.label_r2l is not None and len(data_entry.label_r2l) > 0):
                filtered_unlabeled_entries.append(data_entry)
                total_labeled += 1

        self.trainer.unlabeled_norm_factor = total_labeled / len(self.unlabeled_data)

        # interleave labeled/unlabeled data evenly
        return build_interleaved_batches_from_samples(
            self.labeled_data,
            np.array(filtered_unlabeled_entries),
            self.train_batch_size
        )

    def train_dataloader(self):
        return DataLoader(
            CROHMEDataset(
                self.get_interleaved_train_batches(),
                "weak",
                self.unlabeled_strong_augmentation
            ),
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


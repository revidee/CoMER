from typing import Optional, Any, List
from zipfile import ZipFile

from torch.utils.data.dataloader import DataLoader

from comer.datamodules import CROHMESupvervisedDatamodule, CROHMEFixMatchInterleavedDatamodule, Oracle
from comer.datamodules.crohme import build_dataset, extract_data_entries, get_splitted_indices, \
    build_batches_from_samples, DataEntry, BatchTuple
from comer.datamodules.crohme.dataset import CROHMEDataset
from comer.datamodules.crohme.variants.collate import collate_fn, collate_fn_remove_unlabeled


class CROHMEFixMatchOracleDatamodule(CROHMEFixMatchInterleavedDatamodule):
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
                self.labeled_data, self.unlabeled_data = full_train_data[labeled_indices], full_train_data[
                    unlabeled_indices]

                # unlabeled train-split, used in the "pseudo-labeling" step
                # this uses the same batch size as the eval step, since inference requires more VRAM
                # (due to beam-search)
                self.pseudo_labeling_batches = build_batches_from_samples(
                    self.unlabeled_data,
                    self.eval_batch_size
                )

                self.unlabeled_factor = (1 / (1 - self.unlabeled_pct)) - 1

                # initialize the pseudo-labels with empty labels
                self.trainer.unlabeled_pseudo_labels = {}
                for entry in self.unlabeled_data:
                    self.trainer.unlabeled_pseudo_labels[entry.file_name] = []

                # init oracle
                self.trainer.oracle = Oracle(self.unlabeled_data)

                self.val_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size)[0],
                    "",
                    "",
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size)[0],
                    "",
                    "",
                )
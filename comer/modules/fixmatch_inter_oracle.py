from typing import Callable, List

import torch
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher

from comer.datamodules.crohme import Batch, vocab
from comer.datamodules.crohme.batch import MaybePartialLabel
from comer.modules import CoMERFixMatchInterleaved


class CoMERFixMatchOracleInterleaved(CoMERFixMatchInterleaved):
    def unlabeled_full(self, data_fetcher: AbstractDataFetcher,
                       start_batch: Callable, end_batch: Callable, dataloader_idx: int):
        is_iter_data_fetcher = isinstance(data_fetcher, DataLoaderIterDataFetcher)
        fnames: List[str] = []
        pseudo_labels: List[MaybePartialLabel] = []
        batch: Batch

        batch_idx = 0

        with torch.inference_mode():
            while not data_fetcher.done:
                if not is_iter_data_fetcher:
                    batch = next(data_fetcher)
                else:
                    _, batch = next(data_fetcher)
                start_batch(batch, batch_idx)
                batch_idx = batch_idx + 1

                fnames.extend(batch.img_bases)

                pseudo_labels.extend(
                    [
                        (False, vocab.indices2words(h.seq), None)
                        if self.trainer.oracle.confidence_indices(batch.img_bases[i], h.seq)
                           >= self.pseudo_labeling_threshold
                        else (False, [], None) for i, h in enumerate(self.approximate_joint_search(batch.imgs, batch.mask))
                    ]
                )

                end_batch(batch, batch_idx)
        return zip(fnames, pseudo_labels)

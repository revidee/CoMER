import math
from typing import List, Tuple, Iterable, Union, Callable

import torch
import torch.distributed as dist
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher
from torch import optim

from comer.datamodules.crohme import Batch, vocab
from comer.lit_extensions import UnlabeledLightningModule
from comer.modules.supervised import CoMERSupervised
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out)
import numpy as np

class CoMERSelfTraining(CoMERSupervised, UnlabeledLightningModule):

    def __init__(
            self,
            pseudo_labeling_threshold: float,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.pseudo_labeling_threshold = np.log(pseudo_labeling_threshold)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.labels, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.imgs.shape[0])
        return loss

    def validation_step(self, batch: Batch, batch_idx, dataloader_idx):
        if self.current_epoch <= self.trainer.check_val_every_n_epoch:
            fake_loss = torch.tensor(0.0, device=self.device)
            self.log(
                "val_loss",
                fake_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch.imgs.shape[0]
            )
            self.exprate_recorder([[0]], [[1]])
            self.log(
                "val_ExpRate",
                self.exprate_recorder,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch.imgs.shape[0]
            )
            return fake_loss
        return super().validation_step(batch, batch_idx)

    def unlabeled_full(self, data_fetcher: AbstractDataFetcher,
                       start_batch: Callable, end_batch: Callable, dataloader_idx: int):
        is_iter_data_fetcher = isinstance(data_fetcher, DataLoaderIterDataFetcher)
        batch_indices: List[int] = []
        pseudo_labels: List[List[List[str]]] = []
        batch: Batch

        batch_idx = 0

        with torch.no_grad():
            while not data_fetcher.done:
                if not is_iter_data_fetcher:
                    batch = next(data_fetcher)
                else:
                    _, batch = next(data_fetcher)
                start_batch(batch, batch_idx)
                batch_idx = batch_idx + 1

                batch_indices.append(batch.src_idx)
                # Why h.score / 2?
                # h.score is re-weighted by a final scoring run,
                # adding the loss of a reversed target sequence to the normalized log-likelihood of the beam search.
                # By dividing with 2, we average between these to get a kind-of log-likelihood again.
                pseudo_labels.append(
                    [
                        vocab.indices2words(h.seq) if (h.score / 2) >= self.pseudo_labeling_threshold
                        else [] for h in self.approximate_joint_search(batch.imgs, batch.mask)]
                )

                end_batch(batch, batch_idx)
        return zip(batch_indices, pseudo_labels)

    def validation_unlabeled_step_end(self, to_gather: Iterable[Tuple[int, List[List[str]]]]):
        if not hasattr(self.trainer, 'unlabeled_pseudo_labels'):
            print("warn: trainer does not have the pseudo-label state, cannot update pseudo-labels")
            return

        all_gpu_labels: List[Union[None, List[Tuple[int, List[List[str]]]]]] = [None for _ in range(dist.get_world_size())]
        dist.barrier()
        dist.all_gather_object(all_gpu_labels, list(to_gather))
        # update the gpu-local trainer-cache
        for single_gpu_labels in all_gpu_labels:
            if single_gpu_labels is None:
                continue
            for batch_idx, labels in single_gpu_labels:
                self.trainer.unlabeled_pseudo_labels[batch_idx] = labels

from typing import Dict, Callable, List, Union, Tuple, Iterable

import torch
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher

from comer.datamodules.crohme import Batch, vocab
from comer.modules import CoMERSelfTraining
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out)
import torch.distributed as dist


class CoMERFixMatch(CoMERSelfTraining):

    def __init__(
            self,
            lambda_u: float,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def training_step(self, batches: Dict[str, Batch], _):
        tgt, out = to_bi_tgt_out(batches["labeled"].labels, self.device)
        out_hat = self(batches["labeled"].imgs, batches["labeled"].mask, tgt)

        loss = ce_loss(out_hat, out)
        loggable_total_loss = float(loss)

        batch_size = len(batches["labeled"])
        train_batch_size = batch_size

        if "unlabeled" in batches:
            unlabeled_size = len(batches["unlabeled"])
            # the ce_loss is reduced by mean with the amount of pseudo-labeled samples, which passed the threshold.
            # To align with FixMatch, this has to be normalized to all samples, including below-threshold samples.
            norm_fac = (
                    unlabeled_size / batches["unlabeled"].unfiltered_size
            )
            # floor division, if we don't divide cleanly, we will miss some
            steps = int(unlabeled_size // train_batch_size)
            # if there is any remained, add an extra step
            if (unlabeled_size % train_batch_size) != 0:
                steps += 1
            for i in range(steps):
                loss.backward()
                start, end = i * train_batch_size, (i + 1) * train_batch_size
                if end > unlabeled_size:
                    end = unlabeled_size
                labels, imgs, mask = batches["unlabeled"].labels[start:end],\
                    batches["unlabeled"].imgs[start:end],\
                    batches["unlabeled"].mask[start:end]
                tgt, out = to_bi_tgt_out(labels, self.device)
                out_hat = self(imgs, mask, tgt)
                # average with the amount of samples we processed in this iteration vs. total samples
                # e.g. if we have 25 total, and each iter has 5, this will be weighted with 1/5 (5/25)
                loss = ce_loss(out_hat, out) * ((end - start) / unlabeled_size) * norm_fac * self.hparams.lambda_u
                loggable_total_loss += float(loss)
            batch_size += batches["unlabeled"].imgs.shape[0]

        self.log("train_loss", loggable_total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def unlabeled_full(self, data_fetcher: AbstractDataFetcher,
                       start_batch: Callable, end_batch: Callable, dataloader_idx: int):
        is_iter_data_fetcher = isinstance(data_fetcher, DataLoaderIterDataFetcher)
        fnames: List[str] = []
        pseudo_labels: List[List[str]] = []
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
                # Why h.score / 2?
                # h.score is re-weighted by a final scoring run,
                # adding the loss of a reversed target sequence to the normalized log-likelihood of the beam search.
                # By dividing with 2, we average between these to get a kind-of log-likelihood again.
                pseudo_labels.extend(
                    [
                        vocab.indices2words(h.seq) if (h.score / 2) >= self.pseudo_labeling_threshold
                        else [] for h in self.approximate_joint_search(batch.imgs, batch.mask)]
                )

                end_batch(batch, batch_idx)
        return zip(fnames, pseudo_labels)

    def validation_unlabeled_step_end(self, to_gather: Iterable[Tuple[int, List[List[str]]]]):
        if not hasattr(self.trainer, 'unlabeled_pseudo_labels'):
            print("warn: trainer does not have the pseudo-label state, cannot update pseudo-labels")
            return

        all_gpu_labels: List[Union[None, List[Tuple[str, List[str]]]]] = [None for _ in range(dist.get_world_size())]
        dist.barrier()
        dist.all_gather_object(all_gpu_labels, list(to_gather))
        # update the gpu-local trainer-cache
        for single_gpu_labels in all_gpu_labels:
            if single_gpu_labels is None:
                continue
            for fname, label in single_gpu_labels:
                self.trainer.unlabeled_pseudo_labels[fname] = label


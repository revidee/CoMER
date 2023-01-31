import math
from typing import Callable, List, Iterable, Tuple, Union, Dict

import numpy as np
import torch
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher

from comer.datamodules.crohme import Batch, vocab
import torch.distributed as dist

from comer.datamodules.crohme.batch import MaybePartialLabel
from comer.modules import CoMERFixMatchInterleavedTemperatureScaling
from comer.utils.conf_measures import score_bimin


class CoMERFixMatchInterleavedFixedPctTemperatureScaling(CoMERFixMatchInterleavedTemperatureScaling):

    def __init__(self,
                 pseudo_labeling_threshold: float,
                 keep_old_preds: bool,
                 **kwargs):
        super().__init__(pseudo_labeling_threshold=pseudo_labeling_threshold, **kwargs)
        self.pseudo_labeling_threshold = pseudo_labeling_threshold
        self.keep_old_preds = keep_old_preds

    def unlabeled_full(self, data_fetcher: AbstractDataFetcher,
                       start_batch: Callable, end_batch: Callable, dataloader_idx: int):
        is_iter_data_fetcher = isinstance(data_fetcher, DataLoaderIterDataFetcher)
        fnames: List[str] = []
        pseudo_labels: List[Tuple[MaybePartialLabel, float]] = []
        batch: Batch

        invalid_score = torch.tensor(float("-Inf"), device=self.device)

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
                        (self.maybe_partial_label(h), score_bimin(h)) if (len(h.history) > 0)
                            else ([], invalid_score) for h in self.approximate_joint_search(batch.imgs, batch.mask)]
                )

                end_batch(batch, batch_idx)
        return zip(fnames, pseudo_labels)

    def validation_unlabeled_step_end(self, to_gather: Iterable[Tuple[str, List[Tuple[List[str], float]]]]):
        if not hasattr(self.trainer, 'unlabeled_pseudo_labels'):
            print("warn: trainer does not have the pseudo-label state, cannot update pseudo-labels")
            return

        all_gpu_labels: List[Union[None, List[Tuple[str, Tuple[MaybePartialLabel, float]]]]] = [None for _ in range(dist.get_world_size())]
        dist.barrier()
        dist.all_gather_object(all_gpu_labels, list(to_gather))
        # update the gpu-local trainer-cache
        hyps: List[Tuple[str, MaybePartialLabel]] = []
        scores = []
        merged_labels = {}
        for single_gpu_labels in all_gpu_labels:
            if single_gpu_labels is None:
                continue
            for fname, (partial_label, score) in single_gpu_labels:
                if (partial_label[1] is not None and len(partial_label[1]) > 0) or \
                        (partial_label[2] is not None and len(partial_label[2]) > 0):
                    merged_labels[fname] = (partial_label, score)

        for fname, (label, score) in merged_labels.items():
            hyps.append((fname, label))
            scores.append(score)
        indices = torch.argsort(torch.tensor(scores, device=self.device), descending=True)
        self.trainer.unlabeled_pseudo_labels: Dict[str, MaybePartialLabel]
        if not self.keep_old_preds:
            for fname in self.trainer.unlabeled_pseudo_labels.keys():
                self.trainer.unlabeled_pseudo_labels[fname] = (False, [], None)
        for i in range(int(math.ceil(indices.size(0) * self.pseudo_labeling_threshold))):
            fname, label = hyps[i]
            self.trainer.unlabeled_pseudo_labels[fname] = label

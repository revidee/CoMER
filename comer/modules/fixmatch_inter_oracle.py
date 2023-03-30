import logging
import random
from collections import defaultdict
from typing import Callable, List, Tuple, Iterable, Union, Dict

import torch
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher
from torch.utils.tensorboard import SummaryWriter

from comer.datamodules import Oracle
from comer.datamodules.crohme import Batch, vocab
from comer.datamodules.crohme.batch import MaybePartialLabel
from comer.modules import CoMERFixMatchInterleaved
import torch.distributed as dist


class CoMERFixMatchOracleInterleaved(CoMERFixMatchInterleaved):
    def __init__(
            self,
            pseudo_labeling_threshold: float,
            random_variation: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs, pseudo_labeling_threshold=pseudo_labeling_threshold)
        self.save_hyperparameters()
        self.pseudo_labeling_threshold = pseudo_labeling_threshold

    def unlabeled_full(self, data_fetcher: AbstractDataFetcher,
                       start_batch: Callable, end_batch: Callable, dataloader_idx: int):
        is_iter_data_fetcher = isinstance(data_fetcher, DataLoaderIterDataFetcher)
        fnames: List[str] = []
        pseudo_labels: List[MaybePartialLabel] = []
        lev_dists: List[int] = []
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

                for i, h in enumerate(self.approximate_joint_search(batch.imgs, batch.mask)):
                    lev_dist = self.trainer.oracle.levenshtein_indices_nocutoff(
                        batch.img_bases[i], h.seq
                    )
                    if lev_dist <= self.pseudo_labeling_threshold:
                        fnames.append(batch.img_bases[i])
                        pseudo_labels.append((False, vocab.indices2words(h.seq), None))
                        lev_dists.append(lev_dist)

                end_batch(batch, batch_idx)
        return zip(fnames, pseudo_labels, lev_dists)

    def validation_unlabeled_step_end(self, to_gather: Iterable[Tuple[str, MaybePartialLabel, int]]):
        if not hasattr(self.trainer, 'unlabeled_pseudo_labels'):
            logging.warning("trainer does not have the pseudo-label state, cannot update pseudo-labels")
            return

        all_gpu_labels: List[Union[None, List[Tuple[str, MaybePartialLabel, int]]]] = [None for _ in
                                                                                  range(dist.get_world_size())]
        dist.barrier()
        dist.all_gather_object(all_gpu_labels, list(to_gather))
        # update the gpu-local trainer-cache
        total_passed_this_step = 0
        self.try_reset_pseudo_labels()
        do_random_variation = self.hparams['random_variation']
        err_bins: Dict[int, int] = defaultdict(int)
        hitted_files = set()
        for single_gpu_labels in all_gpu_labels:
            if single_gpu_labels is None:
                continue
            for fname, partial_label, lev_dist in single_gpu_labels:
                if ((partial_label[1] is not None and len(partial_label[1]) > 0) or
                        (partial_label[2] is not None and len(partial_label[2]) > 0)) and (fname not in hitted_files):
                    hitted_files.add(fname)
                    total_passed_this_step += 1
                    err_bins[lev_dist] += 1
                    if not do_random_variation:
                        self.trainer.unlabeled_pseudo_labels[fname] = partial_label

        if do_random_variation:
            # select random label from the unlabeled set, get its annotation and alter it
            unlabeled_options = list(self.trainer.oracle.label_dict.items())
            random.shuffle(unlabeled_options)
            next_idx = 0
            VOCAB_MIN_IDX = 3 # pad, sos, eos are 0, 1, 2
            for (errors, count) in err_bins.items():
                for i in range(count):
                    next_fname, next_label = unlabeled_options[next_idx]
                    next_idx += 1
                    next_label = next_label.copy()
                    for _ in range(errors):
                        err_type = random.randint(0, 2)
                        # draw a random position at which to apply some random modification
                        char_idx = random.randint(0, len(next_label) - 1)
                        if (err_type == 0) and (len(next_label) > 1):
                            # remove character
                            next_label = next_label[:char_idx] + next_label[(char_idx + 1):]
                        elif err_type == 1:
                            # add random character
                            next_label.insert(char_idx, vocab.idx2word[random.randint(VOCAB_MIN_IDX, len(vocab) - 1)])
                        else:
                            # swap with random character from the vocab
                            next_label[char_idx] = vocab.idx2word[random.randint(VOCAB_MIN_IDX, len(vocab) - 1)]
                    # add the potentially altered label to the unlabeled set
                    self.trainer.unlabeled_pseudo_labels[next_fname] = (False, next_label, None)

        if self.local_rank == 0 and self.logger is not None:
            self.log_token_and_len_distribution(total_passed_this_step)

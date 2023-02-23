import os
from pathlib import Path
from typing import Dict, Callable, List, Union, Tuple, Iterable

import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher
from torch.utils.tensorboard import SummaryWriter

from comer.datamodules.crohme import Batch, vocab
from comer.datamodules.crohme.batch import MaybePartialLabel
from comer.modules import CoMERSelfTraining
from comer.utils.conf_measures import th_fn_bimin, score_bimin, score_ori, CONF_MEASURES
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out, ExpRateRecorder, Hypothesis)
import torch.distributed as dist

import logging


class CoMERFixMatch(CoMERSelfTraining):

    def __init__(
            self,
            lambda_u: float,
            partial_labeling_enabled: bool = False,
            partial_labeling_only_below_normal_threshold: bool = False,
            partial_labeling_min_conf: float = 0.0,
            partial_labeling_std_fac: float = 1.0,
            partial_labeling_std_fac_fade_conf_exp: float = 0.0,
            keep_old_preds: bool = False,
            conf_fn: str = 'ori',
            **kwargs
    ):
        super().__init__(**kwargs)
        if partial_labeling_min_conf <= 0.0:
            self.partial_labeling_min_conf = float('-Inf')
        else:
            self.partial_labeling_min_conf = np.log(partial_labeling_min_conf)
        self.save_hyperparameters()
        self.unlabeled_exprate_recorder = ExpRateRecorder()
        self.unlabeled_threshold_passing_exprate_recorder = ExpRateRecorder()
        assert conf_fn in CONF_MEASURES
        self.confidence_fn = CONF_MEASURES[conf_fn]

    def training_step(self, batches: Dict[str, Batch], _):
        tgt, out, l2r_indices, r2l_indices = to_bi_tgt_out(batches["labeled"].labels, self.device)
        out_hat = self(batches["labeled"].imgs, batches["labeled"].mask, tgt, l2r_indices, r2l_indices)

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
                labels, imgs, mask = batches["unlabeled"].labels[start:end], \
                    batches["unlabeled"].imgs[start:end], \
                    batches["unlabeled"].mask[start:end]
                tgt, out, l2r_indices, r2l_indices = to_bi_tgt_out(labels, self.device)
                out_hat = self(imgs, mask, tgt, l2r_indices, r2l_indices)
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
                # Why h.score / 2?
                # h.score is re-weighted by a final scoring run,
                # adding the loss of a reversed target sequence to the normalized log-likelihood of the beam search.
                # By dividing with 2, we average between these to get a kind-of log-likelihood again.
                hyps = self.approximate_joint_search(batch.imgs, batch.mask)
                hyps_to_extend: List[MaybePartialLabel] = [
                    self.maybe_partial_label(h) for h in hyps
                ]
                pseudo_labels.extend(hyps_to_extend)

                if hasattr(self.trainer, "oracle"):
                    for idx, h in enumerate(hyps):
                        label = self.trainer.oracle.get_gt_indices(batch.img_bases[idx])
                        self.unlabeled_exprate_recorder.update([h.seq], [label])
                        if th_fn_bimin(h, self.pseudo_labeling_threshold):
                            self.unlabeled_threshold_passing_exprate_recorder.update([h.seq], [label])


                end_batch(batch, batch_idx)
        return zip(fnames, pseudo_labels)

    def try_reset_pseudo_labels(self):
        if not self.hparams.keep_old_preds:
            for fname in self.trainer.unlabeled_pseudo_labels.keys():
                self.trainer.unlabeled_pseudo_labels[fname] = (False, [], None)

    def maybe_partial_label(self,
                            hyp: Hypothesis,
                            conf_fn: Callable[[Hypothesis], float] = None
                            ) -> MaybePartialLabel:
        if conf_fn is None:
            conf_fn = self.confidence_fn
        total_conf = conf_fn(hyp)
        seq_len = len(hyp.seq)
        if seq_len == 0:
            return False, [], None
        seq_as_words = vocab.indices2words(hyp.seq)
        # we are not partial labeling, or we only partial label below the usual threshold, or the seq is too short
        if (not self.hparams.partial_labeling_enabled) or (
           self.hparams.partial_labeling_only_below_normal_threshold and (total_conf >= self.pseudo_labeling_threshold)
        ) or (seq_len == 1):
            return (False, [], None) if (total_conf < self.pseudo_labeling_threshold)\
                else (False, seq_as_words, None)

        # We are partial labeling, check if the min partial label confidence has been reached
        if total_conf < self.partial_labeling_min_conf:
            return False, [], None
        # We are in the valid range for partial labeling
        # do the actual partial label heuristic

        # find the smallest logit, if it is farther than X*std from the mean of the rest, use left/right side of the idx
        # as partial hyps
        avgs = np.array([(hyp.history[i] + hyp.best_rev[i]) / 2 for i in range(len(hyp.seq))])
        idx = np.argmin(avgs)
        masked_avgs = np.ma.array(avgs, mask=False)
        masked_avgs.mask[idx] = True

        m = masked_avgs.mean()
        power = np.dot(masked_avgs, masked_avgs) / masked_avgs.size
        std = np.sqrt(power - m ** 2)
        min_dev = m - avgs[idx]

        std_fac = self.hparams.partial_labeling_std_fac
        if self.hparams.partial_labeling_std_fac_fade_conf_exp != 0.0:
            std_fac *= np.exp(total_conf * self.hparams.partial_labeling_std_fac_fade_conf_exp)

        if m >= self.pseudo_labeling_threshold or (min_dev >= (std * std_fac)):
            # mask it and use l2r / r2l from there
            return True, seq_as_words[:idx], seq_as_words[idx + 1:]
        return False, [], None

    def log_token_and_len_distribution(self, total_passed_this_step):
        token_dist_file = Path(os.path.join(self.logger.log_dir, 'token_dist_per_epoch.csv'))
        if not token_dist_file.is_file():
            token_dist_file.touch(exist_ok=True)
        len_dist_file = Path(os.path.join(self.logger.log_dir, 'len_dist_per_epoch.csv'))
        if not len_dist_file.is_file():
            token_dist_file.touch(exist_ok=True)

        epoch_and_frequencies = np.zeros((1, len(vocab) + 1), dtype=np.int)
        epoch_and_frequencies[0][0] = self.current_epoch
        epoch_and_lens = np.zeros((1, self.hparams['max_len'] + 1), dtype=np.int)
        epoch_and_lens[0][0] = self.current_epoch

        total_passed = 0
        for partial_label in self.trainer.unlabeled_pseudo_labels.values():
            if (partial_label[1] is not None and len(partial_label[1]) > 0) or \
                    (partial_label[2] is not None and len(partial_label[2]) > 0):
                total_passed += 1
            if partial_label[0]:
                if partial_label[1] is not None and len(partial_label[1]) > 0:
                    epoch_and_lens[0][len(partial_label[1])] += 1
                    for token_idx in vocab.words2indices(partial_label[1]):
                        epoch_and_frequencies[0][token_idx + 1] += 1
                if partial_label[2] is not None and len(partial_label[2]) > 0:
                    epoch_and_lens[0][len(partial_label[2])] += 1
                    for token_idx in vocab.words2indices(partial_label[2]):
                        epoch_and_frequencies[0][token_idx + 1] += 1
            elif partial_label[1] is not None and len(partial_label[1]) > 0:
                # bi-dir, s.t. len and each token counts twice
                epoch_and_lens[0][len(partial_label[1])] += 2
                for token_idx in vocab.words2indices(partial_label[1]):
                    epoch_and_frequencies[0][token_idx + 1] += 2
        f = token_dist_file.open('a')
        np.savetxt(f, epoch_and_frequencies, fmt='%.i', delimiter=', ')
        f.close()
        f = len_dist_file.open('a')
        np.savetxt(f, epoch_and_lens, fmt='%.i', delimiter=', ')
        f.close()

        tb_logger: SummaryWriter = self.logger.experiment

        tb_logger.add_scalar(
            "passed_pseudo_labels_total",
            total_passed,
            self.current_epoch
        )
        tb_logger.add_scalar(
            "passed_pseudo_labels_in_epoch",
            total_passed_this_step,
            self.current_epoch
        )
        tb_logger.add_scalar(
            "exprate_pseudo_labels",
            self.unlabeled_exprate_recorder.compute().item(),
            self.current_epoch
        )
        tb_logger.add_scalar(
            "exprate_passed_pseudo_labels",
            self.unlabeled_threshold_passing_exprate_recorder.compute().item(),
            self.current_epoch
        )
        logging.info(f"passed-epoch: {total_passed_this_step}, total: {total_passed}")


    def validation_unlabeled_step_end(self, to_gather: Iterable[Tuple[str, MaybePartialLabel]]):
        if not hasattr(self.trainer, 'unlabeled_pseudo_labels'):
            logging.warning("trainer does not have the pseudo-label state, cannot update pseudo-labels")
            return

        all_gpu_labels: List[Union[None, List[Tuple[str, MaybePartialLabel]]]] = [None for _ in
                                                                                  range(dist.get_world_size())]
        dist.barrier()
        dist.all_gather_object(all_gpu_labels, list(to_gather))
        # update the gpu-local trainer-cache
        total_passed_this_step = 0
        self.try_reset_pseudo_labels()
        for single_gpu_labels in all_gpu_labels:
            if single_gpu_labels is None:
                continue
            for fname, partial_label in single_gpu_labels:
                if (partial_label[1] is not None and len(partial_label[1]) > 0) or \
                        (partial_label[2] is not None and len(partial_label[2]) > 0):
                    total_passed_this_step += 1
                    self.trainer.unlabeled_pseudo_labels[fname] = partial_label
        if self.local_rank == 0 and self.logger is not None:
            self.log_token_and_len_distribution(total_passed_this_step)

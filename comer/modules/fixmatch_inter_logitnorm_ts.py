import logging
from typing import List, Tuple

import torch
from torch import FloatTensor, LongTensor, Tensor

from comer.datamodules.crohme import Batch
from comer.modules import CoMERFixMatchInterleavedTemperatureScaling
from comer.utils.utils import to_bi_tgt_out, ce_logitnorm_loss, Hypothesis
from torch import linalg as LA

class CoMERFixMatchInterleavedLogitNormTempScale(CoMERFixMatchInterleavedTemperatureScaling):

    def __init__(self,
                 logit_norm_temp: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.logit_temp = torch.Tensor([logit_norm_temp])


    def training_step(self, batch: Batch, _):
        contains_unlabeled = batch.unlabeled_start < len(batch)
        # contains_unlabeled = False
        tgt, out, l2r_repeats, r2l_repeats = to_bi_tgt_out(batch.labels, self.device)
        batch_size = tgt.size(0)
        out_hat = self(batch.imgs, batch.mask, tgt, l2r_repeats, r2l_repeats)

        if contains_unlabeled:
            l2r_labeled_mask = l2r_repeats.view(-1).bool()
            l2r_total_labeled = l2r_labeled_mask[:batch.unlabeled_start].count_nonzero()
            l2r_labeled_mask = l2r_labeled_mask[l2r_labeled_mask]
            l2r_labeled_mask[l2r_total_labeled:] = False
            l2r_unlabeled_mask = ~l2r_labeled_mask

            r2l_labeled_mask = r2l_repeats.view(-1).bool()
            r2l_total_labeled = r2l_labeled_mask[:batch.unlabeled_start].count_nonzero()
            r2l_labeled_mask = r2l_labeled_mask[r2l_labeled_mask]
            r2l_labeled_mask[r2l_total_labeled:] = False
            r2l_unlabeled_mask = ~r2l_labeled_mask

            labeled_mask = torch.cat((l2r_labeled_mask, r2l_labeled_mask), dim=0)
            unlabeled_mask = torch.cat((l2r_unlabeled_mask, r2l_unlabeled_mask), dim=0)

            # labeled loss, averaged to the full batch_size
            loss = ce_logitnorm_loss(out_hat[labeled_mask], out[labeled_mask], self.logit_temp) * labeled_mask.count_nonzero() / batch_size
            # + unlabeled loss, normalized by the "mask rate" of the unlabeled data
            # (i.e. % of successfully pseudo-labeled unlabeled samples)
            unlabeled_norm_fac = 1.0
            if hasattr(self.trainer, 'unlabeled_norm_factor'):
                unlabeled_norm_fac = self.trainer.unlabeled_norm_factor
            else:
                logging.warning("WARN: unlabeled norm factor was unset, but is expected to be set before the training begins.")
            loss += ce_logitnorm_loss(out_hat[unlabeled_mask], out[unlabeled_mask], self.logit_temp) \
                    * self.hparams.lambda_u * unlabeled_mask.count_nonzero() / batch_size
        else:
            loss = ce_logitnorm_loss(out_hat, out, self.logit_temp)

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def approximate_joint_search(
            self, img: FloatTensor, mask: LongTensor, use_new: bool = True,
            save_logits: bool = False, debug=False, temperature=None, global_pruning: str = None
    ) -> List[Hypothesis]:
        if temperature is None:
            temperature = self.current_temperature.item()
        hp = dict(self.hparams)
        if global_pruning is None:
            global_pruning = self.hparams['global_pruning_mode']
        if "temperature" in hp:
            del hp["temperature"]
        if "logit_norm_temp" in hp:
            del hp["logit_norm_temp"]
        return self.comer_model.new_beam_search(
            img, mask, **hp, scoring_run=True, bi_dir=True,
            save_logits=save_logits, debug=debug, temperature=temperature, global_pruning=global_pruning
        )

    def process_out_hat(self, out_hat):
        # self.logit_temp = self.logit_temp.to(self.device)
        # return out_hat / (self.logit_temp * (LA.vector_norm(out_hat, dim=-1, keepdim=True) + 1e-7))
        return out_hat

    def validation_step(self, batch: Batch, batch_idx, dataloader_idx=0) -> Tuple[
        Tensor, Tuple[LongTensor, FloatTensor, List[List[int]]]
    ]:
        tgt, out, _, _ = to_bi_tgt_out(batch.labels, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        self.logit_temp = self.logit_temp.to(self.device)
        loss = ce_logitnorm_loss(out_hat, out, self.logit_temp)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.imgs.shape[0]
        )

        hyps = self.approximate_joint_search(batch.imgs, batch.mask, save_logits=True)

        l2r_labels = [label_tuple[1] for label_tuple in batch.labels]

        self.exprate_recorder([h.seq for h in hyps], l2r_labels)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.imgs.shape[0]
        )

        return (
            loss,
            (out, out_hat, l2r_labels)
        )

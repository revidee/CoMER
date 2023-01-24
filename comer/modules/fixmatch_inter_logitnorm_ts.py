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
        batch_size = len(batch)
        # contains_unlabeled = batch.unlabeled_start < batch_size
        contains_unlabeled = False
        tgt, out = to_bi_tgt_out(batch.labels, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)
        self.logit_temp = self.logit_temp.to(self.device)

        if contains_unlabeled:
            labeled_idx = torch.arange(0, batch.unlabeled_start, device=self.device)
            unlabeled_idx = torch.arange(batch.unlabeled_start, batch_size, device=self.device)

            # bi-dir training
            labeled_idx = torch.cat((labeled_idx, labeled_idx + batch_size), dim=0)
            unlabeled_idx = torch.cat((unlabeled_idx, unlabeled_idx + batch_size), dim=0)

            # labeled loss, averaged to the full batch_size
            loss = ce_logitnorm_loss(out_hat[labeled_idx], out[labeled_idx], self.logit_temp) * labeled_idx.size(0) / (batch_size * 2)
            # + unlabeled loss, normalized by the "mask rate" of the unlabeled data
            # (i.e. % of successfully pseudo-labeled unlabeled samples)
            unlabeled_norm_fac = 1.0
            if hasattr(self.trainer, 'unlabeled_norm_factor'):
                unlabeled_norm_fac = self.trainer.unlabeled_norm_factor
            else:
                print("WARN: unlabeled norm factor was unset, but is expected to be set before the training begins.")
            loss += ce_logitnorm_loss(out_hat[unlabeled_idx], out[unlabeled_idx], self.logit_temp) \
                    * self.hparams.lambda_u * unlabeled_idx.size(0) / (batch_size * 2)
        else:
            loss = ce_logitnorm_loss(out_hat, out, self.logit_temp)

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def approximate_joint_search(
            self, img: FloatTensor, mask: LongTensor, use_new: bool = True,
            save_logits: bool = False, debug=False, temperature=None
    ) -> List[Hypothesis]:
        if temperature is None:
            temperature = self.current_temperature.item()
        hp = dict(self.hparams)
        if "temperature" in hp:
            del hp["temperature"]
        if "logit_norm_temp" in hp:
            del hp["logit_norm_temp"]
        return self.comer_model.new_beam_search(
            img, mask, **hp, scoring_run=True, bi_dir=True,
            save_logits=save_logits, debug=debug, temperature=temperature
        )

    def process_out_hat(self, out_hat):
        # self.logit_temp = self.logit_temp.to(self.device)
        # return out_hat / (self.logit_temp * (LA.vector_norm(out_hat, dim=-1, keepdim=True) + 1e-7))
        return out_hat

    def validation_step(self, batch: Batch, batch_idx, dataloader_idx=0) -> Tuple[
        Tensor, Tuple[LongTensor, FloatTensor, List[float], List[List[int]], List[List[int]]]
    ]:
        tgt, out = to_bi_tgt_out(batch.labels, self.device)
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

        self.exprate_recorder([h.seq for h in hyps], batch.labels)
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
            # TODO: finally abstract the hyp conf score function away
            (out, out_hat, [h.score / 2 for h in hyps], [h.seq for h in hyps], batch.labels)
        )

import math
from typing import Callable, List, Iterable, Tuple, Union

import numpy as np
import torch
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher
from torch import optim

from comer.datamodules.crohme import Batch, vocab

from comer.modules.fixmatch_sorted_fixed_pct_temp_scale import CoMERFixMatchInterleavedFixedPctTemperatureScaling
from comer.utils.utils import ce_loss, to_bi_tgt_out, ce_logitnorm_loss


class CoMERFixMatchInterleavedFixedPctTemperatureScalingLogitNorm(CoMERFixMatchInterleavedFixedPctTemperatureScaling):

    def __init__(self,
                 logit_norm_temp: float,
                 monitor: str,
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
        return loss + 0.0 * self.current_temperature

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": self.hparams.monitor,
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
import torch

from comer.datamodules.crohme import Batch
from comer.modules.fixmatch import CoMERFixMatch
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out)


class CoMERFixMatchInterleaved(CoMERFixMatch):
    def training_step(self, batch: Batch, _):
        batch_size = len(batch)
        contains_unlabeled = batch.unlabeled_start < batch_size
        tgt, out = to_bi_tgt_out(batch.labels, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        if contains_unlabeled:
            labeled_idx = torch.arange(0, batch.unlabeled_start, device=self.device)
            unlabeled_idx = torch.arange(batch.unlabeled_start, batch_size, device=self.device)
            # bi-dir training
            labeled_idx = torch.cat((labeled_idx, labeled_idx + batch_size), dim=0)
            unlabeled_idx = torch.cat((unlabeled_idx, unlabeled_idx + batch_size), dim=0)

            # labeled loss, averaged to the full batch_size
            labeled_avg_fac = (batch.unlabeled_start / batch_size)
            loss = ce_loss(out_hat[labeled_idx], out[labeled_idx]) * labeled_avg_fac
            # + unlabeled loss, normalized by the "mask rate" of the unlabeled data
            # (i.e. % of successfully pseudo-labeled unlabeled samples)
            unlabeled_norm_fac = 1.0
            if hasattr(self.trainer, 'unlabeled_norm_factor'):
                unlabeled_norm_fac = self.trainer.unlabeled_norm_factor
            else:
                print("WARN: unlabeled norm factor was unset, but is expected to be set before the training begins.")
            loss += ce_loss(out_hat[unlabeled_idx], out[unlabeled_idx]) \
                        * self.hparams.lambda_u * unlabeled_norm_fac * (1 - labeled_avg_fac)
        else:
            loss = ce_loss(out_hat, out)

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

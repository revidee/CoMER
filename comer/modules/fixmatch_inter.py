import torch

from comer.datamodules.crohme import Batch
from comer.modules.fixmatch import CoMERFixMatch
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out)


class CoMERFixMatchInterleaved(CoMERFixMatch):
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
            loss = ce_loss(out_hat[labeled_mask], out[labeled_mask]) * labeled_mask.count_nonzero() / batch_size
            # + unlabeled loss, normalized by the "mask rate" of the unlabeled data
            # (i.e. % of successfully pseudo-labeled unlabeled samples)
            unlabeled_norm_fac = 1.0
            if hasattr(self.trainer, 'unlabeled_norm_factor'):
                unlabeled_norm_fac = self.trainer.unlabeled_norm_factor
            else:
                print("WARN: unlabeled norm factor was unset, but is expected to be set before the training begins.")
            loss += ce_loss(out_hat[unlabeled_mask], out[unlabeled_mask]) \
                        * self.hparams.lambda_u * unlabeled_mask.count_nonzero() / batch_size
        else:
            loss = ce_loss(out_hat, out)

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

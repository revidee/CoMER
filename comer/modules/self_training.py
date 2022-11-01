from comer.datamodules.crohme import Batch
from .supervised import CoMERSupervised
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out)


class CoMERSelfTraining(CoMERSupervised):
    def training_step(self, batch: Batch, _):
        print("WIP-LOG IS_BATCH_LABELED: ", batch.is_labeled)
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.imgs.shape[0])

        return loss


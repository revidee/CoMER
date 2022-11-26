from typing import Dict

from comer.datamodules.crohme import Batch
from comer.modules import CoMERSelfTraining
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out)


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

        batch_size = batches["labeled"].imgs.shape[0]

        if "unlabeled" in batches:
            tgt, out = to_bi_tgt_out(batches["unlabeled"].labels, self.device)
            out_hat = self(batches["unlabeled"].imgs, batches["unlabeled"].mask, tgt)
            loss += ce_loss(out_hat, out) * self.hparams.lambda_u
            batch_size += batches["unlabeled"].imgs.shape[0]

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss


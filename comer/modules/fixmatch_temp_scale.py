import torch

from comer.datamodules.crohme import Batch
from comer.modules import CoMERFixMatchInterleaved
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out)


class CoMERFixMatchInterleavedTemperatureScaling(CoMERFixMatchInterleaved):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.current_temperature = torch.nn.Parameter(torch.ones(1) * 1.5)


    def validation_step(self, batch: Batch, batch_idx, dataloader_idx):
        tgt, out = to_bi_tgt_out(batch.labels, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
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

        return {
            "loss": super().validation_step(batch, batch_idx)
        }
    def validation_dataloader_end(self, output):
        print(f"validation_dataloader_end {self.local_rank}")
        print(output)

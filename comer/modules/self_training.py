from pytorch_lightning.trainer.progress import BatchProgress
from pytorch_lightning.utilities.fetching import AbstractDataFetcher
from torch import optim

from comer.datamodules.crohme import Batch
from comer.modules.supervised import CoMERSupervised
from comer.lit_extensions import UnlabeledLightningModule
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out)


class CoMERSelfTraining(CoMERSupervised, UnlabeledLightningModule):
    def training_step(self, batch: Batch, _):
        loss = None
        if batch.is_labeled:
            tgt, out = to_bi_tgt_out(batch.labels, self.device)
            out_hat = self(batch.imgs, batch.mask, tgt)

            loss = ce_loss(out_hat, out)
            self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.imgs.shape[0])
        elif self.trainer.current_epoch > 0:
            pass
            # Check what unlabeled images got pseudo-labels and train on them
            # with torch.no_grad():
            #     batch.labels = [h.seq for h in self.approximate_joint_search(batch.imgs, batch.mask)]
            # np_labels = np.array(batch.labels, dtype=np.object)
            # valid_unlabeled_samples = torch.BoolTensor([len(x) > 0 for x in batch.labels])
            #
            # if valid_unlabeled_samples.any():
            #     # We do have "valid" pseudo-labels for self-training
            #     # use them to train as usual
            #     tgt, out = to_bi_tgt_out(np_labels[valid_unlabeled_samples.detach().numpy().astype(bool)].tolist(), self.device)
            #     out_hat = self(batch.imgs[valid_unlabeled_samples], batch.mask[valid_unlabeled_samples], tgt)
            #
            #     loss = ce_loss(out_hat, out)
            #     self.log("train_loss", loss, on_step=False, on_epoch=True,
            #              sync_dist=True, batch_size=valid_unlabeled_samples.count_nonzero())
            #     self.log("train_loss_unl", loss, on_step=False, on_epoch=True,
            #              sync_dist=True, batch_size=valid_unlabeled_samples.count_nonzero())
        return loss

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
            "monitor": "val_ExpRate/dataloader_idx_0",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def validation_step(self, batch: Batch, batch_idx, dataloader_idx):
        return super().validation_step(batch, batch_idx)

    def unlabeled_full(self, data_fetcher: AbstractDataFetcher, batch_progress: BatchProgress, dataloader_idx: int):
        print("unlabeled_full")

    def validation_unlabeled_step_end(self, output):
        print(output)

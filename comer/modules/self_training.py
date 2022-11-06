import time
from typing import List, Tuple, Iterable, Union, Callable

import torch
from pytorch_lightning.trainer.progress import BatchProgress
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher
from torch import optim
import torch.distributed as dist

from comer.datamodules.crohme import Batch, vocab
from comer.modules.supervised import CoMERSupervised
from comer.lit_extensions import UnlabeledLightningModule
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out)


class CoMERSelfTraining(CoMERSupervised, UnlabeledLightningModule):
    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.labels, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.imgs.shape[0])
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

    def unlabeled_full(self, data_fetcher: AbstractDataFetcher,
                       start_batch: Callable, end_batch: Callable, dataloader_idx: int):

        start_time = time.time()
        is_iter_data_fetcher = isinstance(data_fetcher, DataLoaderIterDataFetcher)
        batch_indices: List[int] = []
        pseudo_labels: List[List[List[str]]] = []
        batch: Batch

        batch_idx = 0

        with torch.no_grad():
            while not data_fetcher.done:
                if not is_iter_data_fetcher:
                    batch = next(data_fetcher)
                else:
                    _, batch = next(data_fetcher)
                start_batch(batch, batch_idx)
                batch_idx = batch_idx + 1

                batch_indices.append(batch.src_idx)
                # batch_labels: List[List[str]] = []
                # for i in range(batch.imgs.size(0)):
                #     batch_labels.append(
                #         vocab.indices2words(
                #             self.approximate_joint_search(batch.imgs[i:i + 1], batch.mask[i:i + 1])[0].seq
                #         )
                #     )
                # pseudo_labels.append(batch_labels)
                pseudo_labels.append(
                    [vocab.indices2words(h.seq) for h in self.approximate_joint_search(batch.imgs, batch.mask)]
                )

                end_batch(batch, batch_idx)

        print(f"pseudo_time[{self.global_rank}]: {time.time() - start_time}")
        return zip(batch_indices, pseudo_labels)

    def validation_unlabeled_step_end(self, to_gather: Iterable[Tuple[int, List[List[str]]]]):
        if not hasattr(self.trainer, 'unlabeled_pseudo_labels'):
            print("warn: trainer does not have the pseudo-label state, cannot update pseudo-labels")
            return

        all_gpu_labels: List[Union[None, List[Tuple[int, List[List[str]]]]]] = [None for _ in range(dist.get_world_size())]
        dist.barrier()
        dist.all_gather_object(all_gpu_labels, list(to_gather))
        # update the gpu-local trainer-cache
        for single_gpu_labels in all_gpu_labels:
            if single_gpu_labels is None:
                continue
            for batch_idx, labels in single_gpu_labels:
                self.trainer.unlabeled_pseudo_labels[batch_idx] = labels

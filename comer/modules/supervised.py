import logging
import math
import zipfile
from typing import List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.utilities import rank_zero_only
from torch import FloatTensor, LongTensor

from comer.datamodules.crohme import Batch
from comer.model.comer import CoMER
from comer.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss,
                               to_bi_tgt_out)
from comer.datamodules.crohme.vocab import vocab as vocabCROHME
from comer.datamodules.hme100k.vocab import vocab as vocabHME

class CoMERSupervised(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        # training
        learning_rate: float,
        learning_rate_target: float = 8e-4,
        steplr_steps: float = 10,
        test_suffix: str = "",

        global_pruning_mode: str = 'sup',
        vocab: str = 'crohme',

        # Backwards compatibility to old checkpoints
        temperature: float = 1.0,
        patience: float = 20,
        monitor: str = 'val_ExpRate',
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["temperature", "patience", "monitor"])

        assert vocab in ['hme', 'crohme']
        if vocab == 'hme':
            self.vocab = vocabHME
        else:
            self.vocab = vocabCROHME

        self.comer_model = CoMER(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            used_vocab=self.vocab
        )

        self.exprate_recorder = ExpRateRecorder()
        self.validation_global_pruning_overwrite = None
    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor,
            l2r_indices: Union[LongTensor, None] = None, r2l_indices: Union[LongTensor, None] = None
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.comer_model(img, img_mask, tgt, l2r_indices, r2l_indices)

    def training_step(self, batch: Batch, _):
        tgt, out, l2r_indices, r2l_indices = to_bi_tgt_out(batch.labels, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt, l2r_indices, r2l_indices)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.imgs.shape[0])

        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out, _, _ = to_bi_tgt_out(batch.labels, self.device)
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

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], [label_tuple[1] for label_tuple in batch.labels])
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.imgs.shape[0]
        )

    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask, global_pruning='none')
        self.exprate_recorder([h.seq for h in hyps], [label_tuple[1] for label_tuple in batch.labels])
        return batch.img_bases, [self.vocab.indices2label(h.seq) for h in hyps], [len(h.seq) for h in hyps], [h.score for h in hyps]

    def test_epoch_end(self, test_outputs: List[Tuple[List[str], List[str], List[int], List[float]]]) -> None:
        exprate = self.exprate_recorder.compute()
        logging.info(f"Validation ExpRate: {exprate}")

        with zipfile.ZipFile(f"result{self.hparams.test_suffix}.zip", "w") as zip_f:
            for img_bases, preds, _, _ in test_outputs:
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)
        with open(f"stats{self.hparams.test_suffix}.txt", "w") as file:
            for img_bases, preds, lens, scores in test_outputs:
                for img_base, pred, length, score in zip(img_bases, preds, lens, scores):
                    file.write(f"{img_base},{pred},{length},{score}\n")
            file.close()
    def approximate_joint_search(
            self, img: FloatTensor, mask: LongTensor, use_new: bool = True,
            save_logits: bool = False, debug=False, temperature=None, global_pruning: str = None
    ) -> List[Hypothesis]:
        if temperature is None:
            temperature = 1
        if self.validation_global_pruning_overwrite is not None:
            global_pruning = self.validation_global_pruning_overwrite
        elif global_pruning is None:
            global_pruning = self.hparams['global_pruning_mode']
        hp = dict(self.hparams)
        if "temperature" in hp:
            del hp["temperature"]
        if use_new:
            return self.comer_model.new_beam_search(img, mask, **hp, scoring_run=True, bi_dir=True,
                                                    save_logits=save_logits, debug=debug,
                                                    temperature=temperature, global_pruning=global_pruning)
        return self.comer_model.beam_search(img, mask, **hp, scoring_run=True, bi_dir=True, debug=debug)


    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate * math.sqrt(self.trainer.world_size),
            momentum=0.9,
            weight_decay=1e-4,
        )

        # reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="max",
        #     factor=0.25,
        #     patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        # )
        gamma = math.exp(math.log(self.hparams.learning_rate_target/self.hparams.learning_rate) / self.hparams.steplr_steps)
        step_size = int(math.ceil(self.trainer.max_epochs / (self.hparams.steplr_steps + 1)))
        step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        scheduler = {
            "scheduler": step_scheduler,
            # "monitor": self.hparams.monitor,
            "interval": "epoch",
            # "frequency": self.trainer.check_val_every_n_epoch,
            "frequency": 1,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    def log_stats_from_dl_suffix(self, suffix: str = ''):
        metrics = self.trainer.callback_metrics
        ts_log_str = ''
        if hasattr(self, 'current_temperature'):
            ts_log_str = f' ts: {self.current_temperature.item():.4f}'
        logging.info(f'Epoch {self.current_epoch}: ExpRate: {metrics[f"val_ExpRate{suffix}"]} loss: {metrics[f"val_loss{suffix}"]}{ts_log_str}')

    @rank_zero_only
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if 'val_ExpRate' in metrics:
            self.log_stats_from_dl_suffix()
        elif 'val_ExpRate/dataloader_idx_0' in metrics:
            self.log_stats_from_dl_suffix('/dataloader_idx_0')

    def validation_dataloader_end(self, output):
        pass

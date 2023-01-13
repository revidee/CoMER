import zipfile
from typing import List, Tuple

import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor

from comer.datamodules.crohme import vocab, Batch
from comer.model.comer import CoMER
from comer.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss,
                               to_bi_tgt_out)


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
        temperature: float,
        # training
        learning_rate: float,
        patience: int,
        test_suffix: str = "",
    ):
        super().__init__()
        self.save_hyperparameters()

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
        )

        self.exprate_recorder = ExpRateRecorder()
    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
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
        return self.comer_model(img, img_mask, tgt)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.labels, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.imgs.shape[0])

        return loss

    def validation_step(self, batch: Batch, _):
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

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.labels)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.imgs.shape[0]
        )

    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.labels)
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps], [len(h.seq) for h in hyps], [h.score for h in hyps]

    def test_epoch_end(self, test_outputs: List[Tuple[List[str], List[str], List[int], List[float]]]) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

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
            save_logits: bool = False, debug=False, temperature=None
    ) -> List[Hypothesis]:
        if temperature is None:
            temperature = 1
        hp = dict(self.hparams)
        del hp["temperature"]
        if use_new:
            return self.comer_model.new_beam_search(img, mask, **hp, scoring_run=True, bi_dir=True, save_logits=save_logits, debug=debug, temperature=temperature)
        return self.comer_model.beam_search(img, mask, **hp, scoring_run=True, bi_dir=True, debug=debug)


    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        # reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="max",
        #     factor=0.25,
        #     patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        # )
        step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.patience, gamma=0.90)
        scheduler = {
            "scheduler": step_scheduler,
            "interval": "epoch",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

from typing import Union, List, Tuple

import torch
from einops import rearrange
from torch import LongTensor, FloatTensor, Tensor, optim

from comer.datamodules.crohme import Batch, vocab
from comer.modules import CoMERFixMatchInterleaved
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out, Hypothesis)

import torch.distributed as dist
import torch.nn.functional as F


class CoMERFixMatchInterleavedTemperatureScaling(CoMERFixMatchInterleaved):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.current_temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def training_step(self, batch: Batch, _):
        # Hack to get a zero-grad (no change) to the current temperature param, which is otherwise not used in training
        return super().training_step(batch, _) + 0.0 * self.current_temperature

    def validation_step(self, batch: Batch, batch_idx, dataloader_idx=0) -> Tuple[Tensor, Tuple[LongTensor, FloatTensor]]:
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

        return (
            loss,
            (out, out_hat)
        )

    def approximate_joint_search(
            self, img: FloatTensor, mask: LongTensor, use_new: bool = True, save_logits: bool = False, debug=False
    ) -> List[Hypothesis]:
        hp = dict(self.hparams)
        del hp["temperature"]
        return self.comer_model.new_beam_search(
            img, mask, **hp, scoring_run=True, bi_dir=True,
            save_logits=save_logits, debug=debug, temperature=self.current_temperature.item()
        )

    def validation_dataloader_end(self, outputs: List[Tuple[Tensor, Tuple[LongTensor, FloatTensor]]]):
        all_gpu_labels: List[Union[None, List[Tuple[Tensor, Tuple[LongTensor, FloatTensor]]]]]\
            = [None for _ in range(dist.get_world_size())] if self.local_rank == 0 else None
        dist.barrier()
        dist.gather_object(outputs, all_gpu_labels)
        # out_hat
        # FloatTensor
        # [2b, l, vocab_size]
        if all_gpu_labels is not None:
            all_labels = []
            all_labels_append = all_labels.append
            all_logits = []
            all_logits_append = all_logits.append
            for single_gpu_outputs in all_gpu_labels:
                for single_gpu_step_output in single_gpu_outputs:
                    flat = rearrange(single_gpu_step_output[1][0], "b l -> (b l)")
                    flat_hat = rearrange(single_gpu_step_output[1][1], "b l e -> (b l) e")
                    all_labels_append(flat.to(self.device))
                    all_logits_append(flat_hat.to(self.device))
            labels = torch.cat(all_labels)
            logits = torch.cat(all_logits)

            # ece_criterion = _ECELoss().to(self.device)

            optimizer = optim.LBFGS([self.current_temperature], lr=0.01, max_iter=50)

            # before_temperature_nll = F.cross_entropy(logits / self.current_temperature, labels,
            #                                          ignore_index=vocab.PAD_IDX, reduction="mean").item()
            # before_temperature_ece = ece_criterion(logits, labels).item()
            #
            # print(f'Before temperature - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}')
            def eval():
                optimizer.zero_grad()
                loss = F.cross_entropy(logits / self.current_temperature, labels,
                                       ignore_index=vocab.PAD_IDX, reduction="mean")
                loss.backward()
                return loss
            self.train(True)
            optimizer.step(eval)
            self.train(False)
            optimizer.zero_grad()

            # after_temperature_nll = F.cross_entropy(logits / self.current_temperature, labels,
            #                                         ignore_index=vocab.PAD_IDX, reduction="mean").item()
            # after_temperature_ece = ece_criterion(logits / self.current_temperature, labels).item()
            # print(f'Optimal temperature: {self.current_temperature.item():.3f}')
            # print(f'After temperature - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}')



class _ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
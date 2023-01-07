from typing import Union, List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, FloatTensor, Tensor, optim

from comer.datamodules.crohme import Batch, vocab
from comer.modules import CoMERFixMatchInterleaved
from comer.utils import ECELoss
from comer.utils.utils import (ce_loss,
                               to_bi_tgt_out, Hypothesis)


class CoMERFixMatchInterleavedTemperatureScaling(CoMERFixMatchInterleaved):

    def __init__(
            self,
            th_optim_correct_weight: float,
            th_optim_sharpening: float,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.current_temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
        self.verbose_temp_scale = False

    def training_step(self, batch: Batch, _):
        # Hack to get a zero-grad (no change) to the current temperature param, which is otherwise not used in training
        return super().training_step(batch, _) + 0.0 * self.current_temperature

    def set_verbose_temp_scale_optim(self, val: bool):
        self.verbose_temp_scale = val

    def validation_step(self, batch: Batch, batch_idx, dataloader_idx=0) -> Tuple[
        Tensor, Tuple[LongTensor, FloatTensor, List[float], List[List[int]], List[List[int]]]
    ]:
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
            # TODO: finally abstract the hyp conf score function away
            (out, out_hat, [h.score / 2 for h in hyps], [h.seq for h in hyps], batch.labels)
        )

    def approximate_joint_search(
            self, img: FloatTensor, mask: LongTensor, use_new: bool = True,
            save_logits: bool = False, debug=False, temperature=None
    ) -> List[Hypothesis]:
        if temperature is None:
            temperature = self.current_temperature.item()
        hp = dict(self.hparams)
        del hp["temperature"]
        return self.comer_model.new_beam_search(
            img, mask, **hp, scoring_run=True, bi_dir=True,
            save_logits=save_logits, debug=debug, temperature=temperature
        )

    def validation_dataloader_end(self, outputs: List[
        Tuple[Tensor, Tuple[LongTensor, FloatTensor, List[float], List[List[int]], List[List[int]]]]
    ]):
        torch.cuda.empty_cache()
        all_gpu_labels: List[Union[None, List[
            Tuple[Tensor, Tuple[LongTensor, FloatTensor, List[float], List[List[int]], List[List[int]]]]
        ]]] = [None for _ in range(dist.get_world_size())] if self.local_rank == 0 else None
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
            correct_scores = []
            correct_scores_append = correct_scores.append
            incorrect_scores = []
            incorrect_scores_append = incorrect_scores.append
            for single_gpu_outputs in all_gpu_labels:
                for single_gpu_step_output in single_gpu_outputs:
                    out, out_hat, scores, seqs, labels = single_gpu_step_output[1]
                    flat = rearrange(out, "b l -> (b l)")
                    flat_hat = rearrange(out_hat, "b l e -> (b l) e")
                    all_labels_append(flat.to(self.device))
                    all_logits_append(flat_hat.to(self.device))
                    for i, label in enumerate(labels):
                        if label == seqs[i]:
                            correct_scores_append(scores[i])
                        else:
                            incorrect_scores_append(scores[i])
            labels = torch.cat(all_labels)
            logits = torch.cat(all_logits)

            correct_scores: Tensor = torch.tensor(correct_scores, device=self.device, requires_grad=True)
            total_correct: Tensor = torch.ones(1, device=self.device, dtype=torch.float,
                                               requires_grad=True) * correct_scores.size(0)
            incorrect_scores: Tensor = torch.tensor(incorrect_scores, device=self.device, requires_grad=True)

            # Optimize Temperature Scaling by minimizing the CE-Loss when scaling the logits
            ece_criterion = ECELoss().to(self.device)
            self.current_temperature = torch.nn.Parameter(torch.ones(1, device=self.device, requires_grad=True))
            self.current_temperature.requires_grad_()

            if self.verbose_temp_scale:
                before_temperature_nll = F.cross_entropy(logits / self.current_temperature, labels,
                                                         ignore_index=vocab.PAD_IDX, reduction="mean").item()
                before_temperature_ece = ece_criterion(logits, labels).item()

                print(f'Before temperature - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}')

            def eval_curr_temp():
                optimizer.zero_grad()
                loss = ece_criterion(logits / self.current_temperature, labels) \
                       + F.cross_entropy(logits / self.current_temperature, labels,
                                         ignore_index=vocab.PAD_IDX, reduction="mean")
                loss.backward()
                return loss

            self.train(True)
            with torch.enable_grad() and torch.inference_mode(False):
                optimizer = optim.LBFGS([self.current_temperature], lr=0.01, max_iter=10000)
                self.comer_model.eval()
                self.comer_model.train(False)
                optimizer.step(eval_curr_temp)
                optimizer.zero_grad()

                if self.verbose_temp_scale:
                    after_temperature_nll = F.cross_entropy(logits / self.current_temperature, labels,
                                                            ignore_index=vocab.PAD_IDX, reduction="mean").item()
                    after_temperature_ece = ece_criterion(logits / self.current_temperature, labels).item()
                    print(f'Optimal temperature: {self.current_temperature.item():.3f}')
                    print(f'After temperature - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}')

                # Find best confidence threshold for pseudo-labeling.
                # TODO: Move from learnable to heuristic, that's why it's currently disabled
                if total_correct > 0 and False:
                    threshold = torch.nn.Parameter(torch.ones(1, device=self.device) * 0.5)
                    optimizer = optim.LBFGS([threshold], lr=0.01, max_iter=1000)

                    def eval_threshold():
                        optimizer.zero_grad()
                        log_th = torch.log(threshold)
                        # corrects = torch.sum((correct_scores >= log_th).float())
                        # corrects = torch.sum((torch.tanh(torch.relu(self.hparams.th_optim_sharpening * (correct_scores - log_th)))))
                        corrects = torch.sum((torch.sigmoid(self.hparams.th_optim_sharpening * (correct_scores - log_th))))
                        # incorrects = torch.sum((incorrect_scores >= log_th).float())
                        # incorrects = torch.sum((torch.tanh(torch.relu(self.hparams.th_optim_sharpening * (incorrect_scores - log_th)))))
                        incorrects = torch.sum(
                            (torch.sigmoid(self.hparams.th_optim_sharpening * (incorrect_scores - log_th))))
                        total_passing = corrects + incorrects
                        #

                        correct_pct = corrects / total_passing if total_passing != 0 else 0.0
                        coverage_pct = corrects / total_correct

                        loss = (1.0 + self.hparams.th_optim_correct_weight) - (
                            ((correct_pct * self.hparams.th_optim_correct_weight) + coverage_pct)
                        )

                        # loss = (total_correct - corrects) + incorrects

                        # print(f"step {loss.item()}, corr: {(corrects / total_passing).item() * 100.0:.1f}"
                        #       f" cov: {(corrects / total_correct).item() * 100.0:.1f}")

                        loss.backward()
                        return loss

                    optimizer.step(eval_threshold)
                    optimizer.zero_grad()

                    if self.verbose_temp_scale:
                        log_th = torch.log(threshold)
                        corrects = torch.sum((correct_scores >= log_th).long())
                        incorrects = torch.sum((incorrect_scores >= log_th).long())
                        total_passing = corrects + incorrects

                        correct_pct = corrects / total_passing if total_passing != 0 else torch.zeros(1, device=self.device)
                        coverage_pct = corrects / total_correct
                        print(f"optim th: {threshold.item():.5f}, corr: {correct_pct.item() * 100.0:.1f}"
                              f" cov: {coverage_pct.item() * 100.0:.1f}")
            self.train(False)

        # sync this manual param optimization across all gpus.
        # since otherwise, this would only be synced at the start of the next forward pass.
        # to ensure all computations until then are consistent, we shall do an extra sync here.
        all_gpu_temp: List[Tensor] = [torch.zeros(1, dtype=torch.float, device=self.device) for _ in range(dist.get_world_size())]
        dist.barrier()
        dist.all_gather(all_gpu_temp, torch.ones(1, device=self.device) * self.current_temperature.item())
        self.current_temperature = torch.nn.Parameter(
            torch.ones(1, device=self.device) * all_gpu_temp[0].to(self.device)
        )





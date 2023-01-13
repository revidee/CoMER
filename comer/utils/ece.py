import itertools
import math
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F


class ECELoss(torch.nn.Module):
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
    def __init__(self, n_bins=15, conf_range=(0.0, 1.0)):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(conf_range[0], conf_range[1], n_bins + 1)
        self.n_bins = n_bins
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

        self.conf_range = conf_range
        self.conf_range_width = conf_range[1] - conf_range[0]

        self.bin_sum_confs = []
        self.bin_sum_corr = []
        self.bin_samples = []
        self.reset_predictions()

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

    def reset_predictions(self):
        self.bin_sum_confs = list(itertools.repeat(0.0, self.n_bins))
        self.bin_sum_corr = list(itertools.repeat(0.0, self.n_bins))
        self.bin_samples = list(itertools.repeat(0, self.n_bins))

    def add_predictions(self, predictions: List[Tuple[float, List[int], List[int]]]):
        """
        Adds a list of predictions to the current averages for ECE calculation
        Parameters
        ----------
        predictions (List[Tuple[float, List[int], List[int]]])
            List of Triplets containing
                - Confidence Score
                - Prediction
                - Label
        """
        for (score, pred, label) in predictions:
            if score < self.conf_range[0] or score > self.conf_range[1]:
                continue

            bin_idx = int(math.floor(((score - self.conf_range[0]) / self.conf_range_width) * self.n_bins))
            if bin_idx >= self.n_bins:
                bin_idx = self.n_bins - 1
            self.bin_sum_confs[bin_idx] += score
            self.bin_sum_corr[bin_idx] += int(pred == label)
            self.bin_samples[bin_idx] += 1

    def ece_for_current_predictions(self):
        ece = 0.0
        total_samples = 0
        total_corr = 0
        for bin_idx, (n_samples, sum_corr, sum_conf) in enumerate(zip(self.bin_samples, self.bin_sum_corr, self.bin_sum_confs)):
            accuracy_in_bin = sum_corr / n_samples if n_samples > 0 else 0.0
            avg_confidence_in_bin = sum_conf / n_samples if n_samples > 0 else 0.0
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * n_samples
            total_corr += sum_corr
            total_samples += n_samples

        return ece / total_samples if total_samples > 0 else 0,\
            total_corr / total_samples if total_samples > 0 else 0

    def get_plot_bins(self, predictions: Union[None, List[Tuple[float, List[int], List[int]]]]=None) -> Tuple[List[float], List[float], List[int]]:
        if predictions is not None:
            self.reset_predictions()
            self.add_predictions(predictions)
        return [
            sum_corr / n_samples if n_samples > 0 else 0.0 for (n_samples, sum_corr, sum_conf) in zip(self.bin_samples, self.bin_sum_corr, self.bin_sum_confs)
        ], [
            sum_conf / n_samples if n_samples > 0 else 0.0 for (n_samples, sum_corr, sum_conf) in zip(self.bin_samples, self.bin_sum_corr, self.bin_sum_confs)
        ], self.bin_samples, self.ece_for_current_predictions()

    def ece_for_predictions(self, predictions: List[Tuple[float, List[int], List[int]]]):
        self.reset_predictions()
        self.add_predictions(predictions)
        ece = self.ece_for_current_predictions()
        self.reset_predictions()
        return ece

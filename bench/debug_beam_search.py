import itertools
import math
import os
import random
from collections import defaultdict
from typing import Dict, Tuple, List, Callable, Union, Any
from zipfile import ZipFile

from PIL import Image
import matplotlib
import numpy as np
import sklearn.metrics as metrics
import torch
from jsonargparse import CLI
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything

from comer.datamodules import Oracle
from comer.datamodules.crohme import extract_data_entries, vocab, DataEntry
from comer.datamodules.crohme.batch import build_batches_from_samples, Batch, get_splitted_indices, BatchTuple, \
    build_interleaved_batches_from_samples
from comer.datamodules.crohme.variants.collate import collate_fn
from comer.datamodules.oracle import general_levenshtein
from comer.datamodules.utils.randaug import RandAugment
from comer.modules import CoMERFixMatchInterleavedTemperatureScaling, CoMERSupervised, \
    CoMERFixMatchInterleavedLogitNormTempScale
from comer.utils import ECELoss
from comer.utils.conf_measures import th_fn_bimin, score_ori, score_bimin, score_avg, score_rev_avg, score_bisum, \
    score_bisum_avg, score_bi_avg, score_sum, score_min
from comer.utils.utils import Hypothesis
from operator import ne
from itertools import compress, count

from model_lookups import POSSIBLE_CP_SHORTCUTS

# checkpoint_path = "./bench/epoch3.ckpt"
checkpoint_path = './lightning_logs/version_48/checkpoints/ep=251-st=51982-valLoss=0.3578.ckpt'

use_fn = th_fn_bimin


def calc_min(th_pseudo_perc: float, all_hyps: Dict[str, Hypothesis], oracle: Oracle):
    th_min = math.log(th_pseudo_perc)
    min_conf_passed = 0
    min_conf_passed_correct = 0
    min_conf_lev_sum = 0
    correct_hyps = 0

    for fname, hyp in all_hyps.items():
        lev_dist = oracle.levenshtein_indices(fname, hyp.seq)
        if lev_dist == 0:
            correct_hyps += 1

        if use_fn(hyp, th_min):
            min_conf_passed += 1
            if lev_dist == 0:
                min_conf_passed_correct += 1
            else:
                min_conf_lev_sum += lev_dist
        # if len(hyp.history) > 0:
        #     min_score = min(hyp.history)
        #     if (min_score * (exp ** (1 + 5 * math.log(seq_len)))) > th_min:
        #         # if (min_score) > th_min:
        #         min_conf_passed += 1
        #         if lev_dist == 0:
        #             min_conf_passed_correct += 1
        #         else:
        #             min_conf_lev_sum += lev_dist
    return min_conf_passed, min_conf_passed_correct, \
        min_conf_passed_correct / correct_hyps if correct_hyps > 0 else 0, \
        min_conf_passed_correct / min_conf_passed if min_conf_passed > 0 else 0


def single_step(input_tuple: Tuple[float, Dict[str, Hypothesis], Oracle]):
    return calc_min(input_tuple[0], input_tuple[1], input_tuple[2], 1.0), input_tuple[0]


def to_rounded_exp(logits: List[float]):
    return [f"{math.exp(logit) * 100:.2f}" for logit in logits]


def main(gpu: int = -1):
    print("init")
    print(f"- gpu: cuda:{gpu}")
    print(f"- cp: {checkpoint_path}")
    if gpu == -1:
        device = torch.device(f"cpu")
    else:
        device = torch.device(f"cuda:{gpu}")

    with torch.no_grad():
        # model: CoMERFixMatchInterleavedTemperatureScaling\
        #     = CoMERFixMatchInterleavedTemperatureScaling.load_from_checkpoint(POSSIBLE_CP_SHORTCUTS['syn_15'], global_pruning_mode="none")
        # model = model.eval().to(device)
        # model.share_memory()
        #
        # print("model loaded")

        with ZipFile("data.zip") as archive:
            seed_everything(7)
            full_data: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "train",
                                                                                     to_device=device)
            # # labeled_indices, unlabeled_indices = get_splitted_indices(
            # #     full_data,
            # #     unlabeled_pct=0.65,
            # #     sorting_mode=1
            # # )
            full_data_test: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "2019")
            # #
            test_batches = build_batches_from_samples(
                full_data_test,
                1
            )

            # augment = RandAugment(3)
            # original = Image.fromarray(test_batches[240][1][0])
            # original.save("augment_original.png")
            # random.seed(231)
            # for i in range(20):
            #     augment(original).save(f"augment{i}.png")
            #
            # exit()

            # batches = build_interleaved_batches_from_samples(
            #     full_data[labeled_indices],
            #     full_data[unlabeled_indices],
            #     4
            # )

            # batch_tuple: List[BatchTuple] = build_batches_from_samples(
            #     entries,
            #     4,
            #     batch_imagesize=(2200 * 250 * 4),
            #     max_imagesize=(2200 * 250),
            #     include_last_only_full=True
            # )[::-1]
            #
            # generate_hyps(
            #     [
            #         #     (
            #         #         "./lightning_logs/version_64/checkpoints/optimized_ts_0.5146.ckpt",
            #         #         CoMERFixMatchInterleavedLogitNormTempScale,
            #         #         "hyps_s_ln35_new_original_t002"
            #         #     ),
            #         # (
            #         #     "./lightning_logs/version_65/checkpoints/optimized_ts_0.5505.ckpt",
            #         #     CoMERFixMatchInterleavedLogitNormTempScale,
            #         #     "hyps_s_ln35_new_original_t005"
            #         # ),
            #         #     (
            #         #     "./lightning_logs/version_66/checkpoints/optimized_ts_0.5421.ckpt",
            #         #     CoMERFixMatchInterleavedLogitNormTempScale,
            #         #     "hyps_s_ln35_new_original_t01"
            #         # ),
            #         # (
            #         #     "./lightning_logs/version_67/checkpoints/optimized_ts_0.5038.ckpt",
            #         #     CoMERFixMatchInterleavedLogitNormTempScale,
            #         #     "hyps_s_ln35_new_original_t02"
            #         # ), (
            #         #     "./lightning_logs/version_68/checkpoints/optimized_ts_0.4829.ckpt",
            #         #     CoMERFixMatchInterleavedLogitNormTempScale,
            #         #     "hyps_s_ln35_new_original_t05"
            #         # ), (
            #         #     "./lightning_logs/version_69/checkpoints/optimized_ts_0.556297.ckpt",
            #         #     CoMERFixMatchInterleavedLogitNormTempScale,
            #         #     "hyps_s_ln35_new_original_t0075"
            #         # ), (
            #         #     "./lightning_logs/version_70/checkpoints/optimized_ts_0.5405.ckpt",
            #         #     CoMERFixMatchInterleavedLogitNormTempScale,
            #         #     "hyps_s_ln35_new_original_t004"
            #         # ),
            #         # (
            #         #     "./lightning_logs/version_71/checkpoints/optimized_ts_0.5338.ckpt",
            #         #     CoMERFixMatchInterleavedLogitNormTempScale,
            #         #     "hyps_s_ln35_new_original_t00625"
            #         # ),
            #         (
            #             "./lightning_logs/version_16/checkpoints/epoch=275-step=209484-val_ExpRate=0.5947.ckpt",
            #             CoMERSupervised,
            #             "hyps_s_50_new_original"
            #         ),
            #     ],
            #     [
            #         ('_test', test_batches)
            #     ],
            #     device,
            #     [1]
            # )
            # oracle = Oracle(full_data_test)
            # oracle = Oracle(full_data)
            # oracle.add_data(full_data_test)
            # #
            # confidence_measure_ap_ece_table(
            #     [
            #         # torch.load("../hyps_s_ln35_new_original_t00625_1_test.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_ln35_new_original_t00625_test.pt", map_location=torch.device('cpu')),
            #         torch.load("../hyps_s_ln35_new_original_t002_test.pt", map_location=torch.device('cpu')),
            #         torch.load("../hyps_s_ln35_new_original_t004_test.pt", map_location=torch.device('cpu')),
            #         torch.load("../hyps_s_ln35_new_original_t005_test.pt", map_location=torch.device('cpu')),
            #         torch.load("../hyps_s_ln35_new_original_t00625_test.pt", map_location=torch.device('cpu')),
            #         torch.load("../hyps_s_ln35_new_original_t0075_test.pt", map_location=torch.device('cpu')),
            #         torch.load("../hyps_s_ln35_new_original_t01_test.pt", map_location=torch.device('cpu')),
            #         torch.load("../hyps_s_ln35_new_original_t02_test.pt", map_location=torch.device('cpu')),
            #         torch.load("../hyps_s_ln35_new_original_t05_test.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_ln35_new_original_t002_test.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_35_new_original_1.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_35_new_original_ts_ce.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_35_new_original_ts_ece.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_35_new_original_ts_both.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_35_new_original_1_test.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_35_new_original_ts_ce_test.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_35_new_original_ts_ece_test.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_35_new_original_ts_both_test.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_100_new_original_test.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_35_new_original_1_test.pt", map_location=torch.device('cpu')),
            #         # torch.load("../hyps_s_15_new_original_test.pt", map_location=torch.device('cpu')),
            #     ],
            #     oracle
            # )

            # eval_sorting_score(oracle, True, 1.0, True)

            font_size = 42
            dpi = 96
            # dpi = 120
            # fig_size_px = (260*4, 250*4)
            fig_size_px = (1920, 1080)
            matplotlib.rcParams.update({'font.size': font_size})
            # fig, axes = plt.subplots(1, figsize=(fig_size_px[0] / dpi, fig_size_px[1] / dpi))
            # fig.tight_layout(pad=1.3)

            # hyps_base = torch.load("../hyps_s_35_new_original_1_test.pt", map_location=torch.device('cpu'))
            # # # hyps_base = torch.load("../hyps_s_35_new_original_ts_ce_test.pt", map_location=torch.device('cpu'))
            # # # hyps_base = torch.load("../hyps_s_35_new_original_1.pt", map_location=torch.device('cpu'))
            # # # hyps_base = torch.load("../hyps_s_35_new_original_ts_ece.pt", map_location=torch.device('cpu'))
            # # # hyps_ln = torch.load("../hyps_s_35_new_t0_1_opt.pt", map_location=torch.device('cpu'))
            # #
            # calc_tp_as_all_correct = True
            # # #
            # # # # BIMIN, 3.5, partial 0
            # # #
            # for (all_hyps, name, fn, partial_mode, fac, threshold, min_threshold, color) in [
            #     # (hyps_base, "ORI", score_ori, 5, 3.5, 0.5, 0.0, matplotlib.colormaps['Greens'](0.6)),
            #     (hyps_base, "ORI", score_ori, 0, None, 0.0, 0.0, matplotlib.colormaps['Greens'](0.85)),
            #     # (hyps_base, "BIMIN (Partial)", score_bimin, 0, 0.0, 0.6, 0.0, matplotlib.colormaps['Reds'](0.6)),
            #     (hyps_base, "BIMIN", score_bimin, 0, None, 0.0, 0.0, matplotlib.colormaps['Reds'](0.6)),
            #     # (hyps_base, "BIMULT", score_bisum, 5, 3.5, 0.5, 0.0, matplotlib.colormaps['Blues'](0.6)),
            #     (hyps_base, "BIMULT", score_bisum, 0, None, 0.0, 0.0, matplotlib.colormaps['Blues'](0.6)),
            #     # (hyps_ln, "ORI LN", score_ori, 0, None, 0.0, 0.0, matplotlib.colormaps['Blues'](0.9)),
            #     # (hyps_ln, "ORI LN", score_ori, 0, 3.5, 0.93, 0.15, matplotlib.colormaps['Blues'](0.7)),
            #     # (hyps_ln, "ORI LN", score_ori, 5, 3.5, 0.93, 0.15, matplotlib.colormaps['Blues'](0.6)),
            #     # (hyps_ln, "ORI LN", score_ori, 6, 3.5, 0.93, 0.15, matplotlib.colormaps['Blues'](0.5)),
            #     # (hyps_ln, "ORI LN", score_ori, 7, 3.5, 0.93, 0.15, matplotlib.colormaps['Blues'](0.3)),
            #     # ("BIMIN", score_bimin, 3, 0.8, matplotlib.colormaps['Greens']),
            #     # ("BIAVG", score_bi_avg, 0, 0.6, matplotlib.colormaps['Greens']),
            #     # ("BIMIN", score_bimin, 0, 0.6, matplotlib.colormaps['Blues']),
            # ]:
            #     threshold = (float('-Inf') if threshold == 0.0 else np.log(threshold))
            #     min_threshold = float('-Inf') if min_threshold == 0.0 else np.log(min_threshold)
            #     precisions, recalls, auc, skips, total = average_precision(
            #         all_hyps,
            #         fn,
            #         oracle,
            #         fac,
            #         calc_tp_as_all_correct=calc_tp_as_all_correct,
            #         partial_mode=partial_mode,
            #         partial_threshold=threshold,
            #         min_threshold=min_threshold
            #     )
            #     visual = metrics.PrecisionRecallDisplay(precisions, recalls)
            #     # visual.plot(ax=axes, name=f"{name} ({partial_mode}) {fac} ({skips} {zero_safe_division(skips*100, total):.1f}) AP={auc*100:.2f}", color=colormap(color_range[0] - ((color_range[0] - color_range[1]) * zero_safe_division(fac_i, len(facs) - 1))))
            #     # visual.plot(ax=axes, name=f"{name} ({partial_mode}) {fac} ({skips} {zero_safe_division(skips*100, total):.1f}) AP={auc*100:.2f}, CORR={auc*100/max(recalls):.2f}", color=color)
            #     visual.plot(ax=axes, name=f"{name}, AP={auc*100:.1f}", color=color, linewidth=4.5)
            #
            # axes.set_title("35%")
            # # plt.show()
            # # axes[1].set_ylim((0.52, 1.03))
            # axes.set_xlim((0.0, 1.0))
            # plt.savefig("ap_ori_bimin_bimult_35_t1_all_corr.pdf", format="pdf")


            # token_scores = np.zeros((len(vocab),))
            # full_data_test: List[DataEntry]
            # for test_entry in full_data:
            #     for token in vocab.words2indices(test_entry.label):
            #         token_scores[token] += 1
            #
            # gt_without_brackets = token_scores[3:-3] / token_scores[3:].sum()
            #
            # order = np.argsort(gt_without_brackets)[::-1]
            # #
            # fig, axes = plt.subplots(1, figsize=(fig_size_px[0] / dpi, fig_size_px[1] / dpi))
            # fig.tight_layout(pad=0.1)
            # plt.gcf().subplots_adjust(left=0.09, top=0.975, right=0.975, bottom=0.127)
            # def plot_token_dist(ax, dist, col, label="_", lw: float=2):
            #     lines = axes.step(range(len(dist)), dist[order], where="post")
            #     # lines = ax.step(range(len(dist)), dist, where="post")
            #     lines[0].set_label(label)
            #     for l in lines:
            #         l.set(color=col, linewidth=lw)
            #         # l = matplotlib.lines.Line2D([i, i+1], [dist[source_idx], dist[source_idx]], color=col, linewidth=lw)
            #         # axes.add_line(l)
            #         pass
            #
            # def plot_token_dist2(ax, dist, col):
            #     for i, source_idx in enumerate(order):
            #     # for i, am in enumerate(dist):
            #         ax.bar(i, dist[source_idx],
            #         # ax.bar(i, am,
            #                color=col, width=1, align='edge', edgecolor="none")
            # #
            # #
            # #
            # #
            # plot_token_dist2(axes, gt_without_brackets, "#aaaaaa")
            # # # plot_token_dist(axes, token_scores[3:], "#66666677")
            # #
            # axes.set_xlim((0, len(vocab) - 6))
            # axes.set_yscale("log")
            # axes.set_ylim((0, 0.05))
            # axes.set_xticks([])
            # axes.set_xticklabels([])
            # axes.set_xlabel("CROHME Alphabet")
            # axes.set_ylabel("Rel. Vorkommen")
            # #
            # base_path = "A:\Masterabeit\checkpoints"
            # versions = [248, 249, 250, 251, 243, 247, 245, 246]
            # # versions = [214, 215, 216, 217]
            # versions_labels = ["0+Var", "1+Var", "2+Var", "3+Var"]
            # #
            # LAST_EPOCHS = 5
            #
            # with open(f"gt.data", "w") as f:
            #     f.write(f"a;val\n")
            #     for i, val in enumerate((gt_without_brackets)[order]):
            #         f.write(f"{i};{val}\n")
            #
            # for i, version in enumerate(versions):
            #     token_dist = np.loadtxt(os.path.join(base_path, f"version_{version}", "token_dist_per_epoch.csv"), dtype=float, delimiter=",")
            #     len_dist = np.loadtxt(os.path.join(base_path, f"version_{version}", "len_dist_per_epoch.csv"), dtype=float, delimiter=",")
            #     print((token_dist[-LAST_EPOCHS:, 4:-3].mean(axis=0) / token_dist[-LAST_EPOCHS:, 4:].mean(axis=0).sum())[order][50:70],)
            #     with open(f"version_{version}.data", "w") as f:
            #         f.write(f"a;val\n")
            #         for i, val in enumerate((token_dist[-LAST_EPOCHS:, 4:-3].mean(axis=0) / token_dist[-LAST_EPOCHS:, 4:].mean(axis=0).sum())[order]):
            #             f.write(f"{i};{val}\n")
                # np.savetxt(f"versoin_{version}.data", (token_dist[-LAST_EPOCHS:, 4:-3].mean(axis=0) / token_dist[-LAST_EPOCHS:, 4:].mean(axis=0).sum())[order],
                #            delimiter=",")
                # plot_token_dist(axes,
                #                 token_dist[-LAST_EPOCHS:, 4:-3].mean(axis=0) / token_dist[-LAST_EPOCHS:, 4:].mean(axis=0).sum(),
                #                 matplotlib.colormaps['Blues'](((i+1) / (len(versions)) * 0.7 + 0.2)), lw=1.5, label=versions_labels[i]
                #                 )
            # # get the legend object
            # leg = axes.legend(loc="lower left")
            #
            # # change the line width for the legend
            # for line in leg.get_lines():
            #     line.set_linewidth(5.0)
            # # plt.savefig("token_dist_35_ora_var.pdf", format="pdf")
            # plt.show()


            # versions = [248, 249, 250, 251]
            # versions = [208, 209, 211, 213]
            # versions_labels = ["0", "1", "2", "3"]
            # for i, version in enumerate(versions):
            #     token_dist = np.loadtxt(os.path.join(base_path, f"version_{version}", "token_dist_per_epoch.csv"), dtype=float, delimiter=",")
            #     len_dist = np.loadtxt(os.path.join(base_path, f"version_{version}", "len_dist_per_epoch.csv"), dtype=float, delimiter=",")
            #
            #     plot_token_dist(axes,
            #                     token_dist[-LAST_EPOCHS:, 4:-3].mean(axis=0) / token_dist[-LAST_EPOCHS:, 4:].mean(axis=0).sum(),
            #                     matplotlib.colormaps['Reds'](((i+1) / (len(versions)) * 0.7 + 0.2)), lw=1.5, label=versions_labels[i]
            #                     )
            # # get the legend object
            # leg = axes.legend(loc="lower left")
            #
            # # change the line width for the legend
            # for line in leg.get_lines():
            #     line.set_linewidth(5.0)
            # plt.savefig("token_dist_35_ora_novar.pdf", format="pdf")


            # LEN COMP
            syn_15_corr_lens = np.array([0, 0, 31, 196, 75, 108, 119, 88, 34, 72, 81, 80, 63, 52, 42, 59, 26, 45, 36, 40, 22, 30, 24, 35, 23, 28, 20, 20, 10, 14, 7, 8, 3, 10, 7, 5, 3, 6, 6, 5, 3, 8, 0, 4, 1, 3, 2, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 2, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            s_15_corr_lens = np.array([0, 0, 29, 181, 66, 96, 108, 82, 28, 65, 70, 64, 45, 47, 37, 52, 20, 40, 27, 30, 17, 25, 18, 27, 16, 26, 13, 12, 8, 13, 4, 4, 5, 5, 4, 4, 1, 1, 1, 0, 1, 5, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            gt_2019_len = np.array([0, 1, 43, 297, 110, 158, 190, 142, 60, 115, 136, 139, 111, 120, 85, 147, 71, 95, 95, 101, 62, 83, 65, 94, 68, 88, 60, 55, 39, 40, 35, 24, 19, 32, 27, 40, 19, 28, 16, 17, 17, 26, 8, 13, 9, 9, 11, 5, 8, 8, 5, 13, 4, 8, 7, 5, 2, 4
                                       , 4, 4, 2, 2, 3, 5, 1, 1, 1, 1, 3, 1, 3, 0, 1, 1, 0, 3, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                       , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            group = 5
            gt_2019_len = np.add.reduceat(gt_2019_len, np.arange(0, len(gt_2019_len), group))
            s_15_corr_lens = np.add.reduceat(s_15_corr_lens, np.arange(0, len(s_15_corr_lens), group))
            syn_15_corr_lens = np.add.reduceat(syn_15_corr_lens, np.arange(0, len(syn_15_corr_lens), group))



            # np.nan_to_num(correct_lens / total_lens)
            # plot_token_dist2(axes, gt_2019_len, "#aaaaaa")
            plot_token_dist(axes, np.nan_to_num(syn_15_corr_lens / gt_2019_len), matplotlib.colormaps['Greens'](0.65), label="Syn. 15%", lw=4.5)
            plot_token_dist(axes, np.nan_to_num(s_15_corr_lens / gt_2019_len), matplotlib.colormaps['Reds'](0.65), label="15%", lw=4.5)




            # axes.set_yscale("log")
            # axes.set_ylim((0, 1.0))
            axes.set_xlim((0, 15))
            axes.set_ylim((0, 1.0))
            axes.set_xlabel("Seq. Länge")
            axes.set_ylabel("Rel. Korrekt")

            axes.set_xticks([i * 3 for i in range(6)])
            axes.set_xticklabels([i * 3 * group for i in range(6)])

            # ax2 = axes.twinx()
            # ax2.set_ylim((0.0, 1.0))
            # plot_token_dist(ax2, np.nan_to_num(syn_15_corr_lens / gt_2019_len), matplotlib.colormaps['Greens'](0.65))
            # plot_token_dist(ax2, np.nan_to_num(s_15_corr_lens / gt_2019_len), matplotlib.colormaps['Reds'](0.65))

            leg = axes.legend(loc="upper right")

            # change the line width for the legend
            for line in leg.get_lines():
                line.set_linewidth(5.0)
            # plt.show()
            plt.savefig("len_correct_comp_grouped5_alltest.pdf", format="pdf")


            # fig, axes = plt.subplots(1, figsize=(fig_size_px[0] / dpi, fig_size_px[1] / dpi))
            # fig.tight_layout(pad=1.8)
            # plt.gcf().subplots_adjust(left=0.2, top=0.9)
            # lw = 4.5
            #
            # x = [0.02, 0.04, 0.05, 0.0625, 0.075, 0.1]
            # ece = [7.5, 5.4, 6.6, 4.5, 7.7, 16.4]
            # expr = [51.209, 53.711, 54.796, 54.918, 55.213, 54.045]
            #
            # axes.set_xlim((0.02, 0.1))
            # axes.set_xticks([0.02, 0.05, 0.075, 0.1])
            # axes.set_ylabel("ECE (MIN)", color=matplotlib.colormaps['Reds'](0.85))
            # axes.tick_params(axis='y', labelcolor=matplotlib.colormaps['Reds'](0.85))
            # axes.set_xlabel("LogitNorm T")
            #
            # axes.plot(x, ece, color=matplotlib.colormaps['Reds'](0.85), linewidth=lw, label='ECE')
            # ax2 = axes.twinx()
            # ax2.set_ylabel("ExpRate-0", color=matplotlib.colormaps['Blues'](0.85))
            # ax2.tick_params(axis='y', labelcolor=matplotlib.colormaps['Blues'](0.85))
            # ax2.plot(x, expr, color=matplotlib.colormaps['Blues'](0.85), linewidth=lw, label='ExpRate-0')
            #
            # plt.savefig("ln_param_search.pdf", format="pdf")
            # exit()

            #
            # x =         [100,   75,     65,     55,     50,     40,     35,     30,     25,     20,     15]
            # exp_14 =    [57.00, 57.81,  56.29,  51.83,  52.53,  49.70,  48.28,  45.94,  43.81,  42.80,  37.12]
            # exp_16 =    [59.98, 58.06,  58.24,  55.71,  53.36,  53.36,  51.00,  48.82,  47.52,  44.99,  39.06]
            # exp_19 =    [63.39, 61.88,  61.05,  60.13,  59.22,  56.30,  54.55,  53.46,  50.46,  48.37,  40.53]
            #
            # oracle_x = [75, 50, 25, 15]
            # oracle_14 = [56.29, 52.94, 48.78, 43.00]
            # oracle_16 = [57.8, 55.45, 50.04, 42.72]
            # oracle_19 = [62.64, 59.47, 54.55, 47.21]
            #
            # axes.plot(x, exp_14, color=matplotlib.colormaps['Reds'](0.85), linewidth=lw, label='2014')
            # axes.plot(x, exp_16, color=matplotlib.colormaps['Blues'](0.85), linewidth=lw, label='2016')
            # axes.plot(x, exp_19, color=matplotlib.colormaps['Greens'](0.85), linewidth=lw, label='2019')
            #
            # # axes.plot(oracle_x, oracle_14, color=matplotlib.colormaps['Reds'](0.45), linewidth=lw, label='FixMatch Oracle 2014')
            # # # axes.plot(oracle_x, oracle_16, color=matplotlib.colormaps['Blues'](0.45), linewidth=lw, label='FixMatch Oracle 2016')
            # # # axes.plot(oracle_x, oracle_19, color=matplotlib.colormaps['Greens'](0.45), linewidth=lw, label='FixMatch Oracle 2019')
            # #
            # axes.legend(loc='lower left')
            # axes.set_xlim((0, 100))
            # axes.set_ylim((0, 65))
            # ticks = [25, 50, 75, 100]
            # axes.set_xticks(ticks)
            # axes.set_xticklabels(
            #     [f"{tick}" for tick in ticks], rotation=0
            #     # , fontdict={'fontfamily': 'Iosevka'}
            # )
            # # for (xpos, ypos) in zip(x, exp_19):
            # #     axes.annotate(f"{ypos:.1f}", (xpos, ypos), fontsize=12)
            # # for (normal, oracle) in zip([exp_14], [oracle_14]):
            # #     for (xpos, ypos) in zip(oracle_x, oracle):
            # #         other_idx = x.index(xpos)
            # #         diff = (ypos - normal[other_idx])
            # #         axes.annotate(f"{'+' if diff > 0 else ''}{diff:.2f}", (xpos, ypos), fontsize=12)
            # # # labels = [f"{samples_in_bin}" for (i, samples_in_bin) in enumerate(samples)]
            # # # labels = [f"{(i+1)/len(accs):.2f})\n{samples_in_bin}" for (i, samples_in_bin) in enumerate(samples)]
            # # # labels[-1] = f"1.0]\n{samples[-1]}"
            #
            # axes.set_xlabel('% Trainingsdaten')
            # axes.set_ylabel('ExpRate 0')
            # axes.grid(linewidth=1)
            # # plt.show()
            # plt.savefig("exprate_all.pdf", format="pdf")


            # Validation Length Plot
            def load_validation_times(path: str):
                return np.loadtxt(path, delimiter=',', dtype=int)
            #
            # vals_342 = load_validation_times("A:/Masterabeit/valtime/val_time_342.csv")
            # vals_343 = load_validation_times("A:/Masterabeit/valtime/val_time_343.csv")
            #
            # axes.plot(vals_342[1:, 0], vals_342[1:, 1], color=matplotlib.colormaps['Greens'](0.85), linewidth=3.5, label='Neu+Constant')
            # axes.plot(vals_343[1:, 0], vals_343[1:, 1], color=matplotlib.colormaps['Reds'](0.85), linewidth=3.5, label='Neu')
            #
            # axes.set_xlabel("Epoche")
            # axes.set_ylabel("Validierungszeit [s]")
            #
            # # axes.annotate(
            # #     f'{vals_343[3][1]}s',
            # #     (2, vals_343[3][1]),
            # #     xytext=(0.1, 0.8),
            # #     textcoords='axes fraction',
            # #     arrowprops=dict(facecolor='black', arrowstyle='wedge'),
            # #     fontsize=24, annotation_clip=False)
            # # axes.annotate(
            # #     f'{vals_343[4][1]}s',
            # #     (3, vals_343[4][1]),
            # #     xytext=(0.15, 0.7),
            # #     textcoords='axes fraction',
            # #     arrowprops=dict(facecolor='black', arrowstyle='wedge'),
            # #     fontsize=24, annotation_clip=False)
            #
            # # axes.set_yscale('log')
            # axes.set_ylim((0, 1000))
            #
            # axes.set_xlim((1, 250))
            #
            # axes.legend()
            #
            # plt.show()



            # hyps_s_35_new_original_1.pt

            # hyps: Dict[str, Hypothesis] = torch.load("../hyps_s_35_new_original_ts_ece.pt",
            #                                              map_location=torch.device('cpu'))
            #
            #
            # # Rates a list of thresholds with correct/incorrect and total/avg levensthein metrics
            #
            # splitted_fnames = []
            # splitted_hyps: List[Hypothesis] = []
            # splitted_scores = []
            #
            # for (fname, hyp) in hyps.items():
            #     splitted_fnames.append(fname)
            #     splitted_hyps.append(hyp)
            #     splitted_scores.append(score_bimin(hyp))
            #
            # splitted_scores = np.array(splitted_scores)
            # splitted_scores_sorted = np.argsort(splitted_scores)[::-1]
            #
            # total_to_print = 10
            # skips = 0
            # for best_i, idx in enumerate(splitted_scores_sorted):
            #     score = splitted_scores[idx]
            #     hyp: Hypothesis = splitted_hyps[idx]
            #     fname: str = splitted_fnames[idx]
            #     lev_dist = oracle.levenshtein_indices(fname, hyp.seq)
            #     if lev_dist != 0:
            #         if skips:
            #             skips -= 1
            #             continue
            #         print("###############")
            #         print(best_i)
            #         print("gt")
            #         print(vocab.indices2words(oracle.get_gt_indices(fname)))
            #         print("pred")
            #         print(vocab.indices2words(hyp.seq))
            #         print(to_rounded_exp(hyp.history))
            #         print(to_rounded_exp(hyp.best_rev))
            #         print("avgs")
            #         avgs = [(hyp.history[i] + hyp.best_rev[i]) / 2 for i in range(len(hyp.seq))]
            #         print(to_rounded_exp(avgs))
            #
            #         print(lev_dist)
            #         print(fname, math.exp(score), hyp.was_l2r, len(hyp.all_l2r_scores), len(hyp.all_r2l_scores))
            #         # print(hyp.all_l2r_hyps)
            #         # print(hyp.all_r2l_hyps)
            #         total_to_print -= 1
            #         if total_to_print == 0:
            #             break
            #         # for i, rev_hyp in enumerate(hyp.all_l2r_hyps):
            #         #     print("### l2r")
            #         #     print(math.exp(hyp.all_l2r_scores[i]))
            #         #     print(vocab.indices2words(rev_hyp.tolist()))
            #         #     print(to_rounded_exp(hyp.all_l2r_history[i].tolist()))
            #         # for i, rev_hyp in enumerate(hyp.all_r2l_hyps):
            #         #     print("### r2l")
            #         #     print(math.exp(hyp.all_r2l_scores[i]))
            #         #     print(vocab.indices2words(rev_hyp.tolist()))
            #         #     print(to_rounded_exp(hyp.all_r2l_history[i].tolist()))

            # for idx in [1, 2, 3, 4, 5, 10, 100]:
            #     if idx != 1:
            #         hyp_file = f"../hyps_st_15_tmp_{idx}.pt"
            #         hyp_file_noglob = f"../hyps_st_15_tmp_{idx}_noglobal.pt"
            #     else:
            #         hyp_file = f"../hyps_st_15.pt"
            #         hyp_file_noglob = ""
            #     ece_score, acc = None, None
            #     ece_score_noglob, acc_noglob = None, None
            #     if len(hyp_file) > 0:
            #         all_hyps: Dict[str, Hypothesis] = torch.load(hyp_file,
            #                                                      map_location=torch.device('cpu'))
            #         ece_score, acc = ece.ece_for_predictions(map(hyp_to_triplet, all_hyps.items()))
            #     if len(hyp_file_noglob) > 0:
            #         all_hyps: Dict[str, Hypothesis] = torch.load(hyp_file_noglob,
            #                                                      map_location=torch.device('cpu'))
            #         ece_score_noglob, acc_noglob = ece.ece_for_predictions(map(hyp_to_triplet, all_hyps.items()))
            #     if ece_score is not None:
            #         print(f"{idx}\t\t{ece_score * 100:.2f} ({acc * 100:.2f})", end="")
            #     if ece_score_noglob is not None:
            #         print(f"\t\t{ece_score_noglob * 100:.2f} ({acc_noglob * 100:.2f})")
            #     else:
            #         print()
            #
            # th = math.log(0.64618)
            #
            # all_hyps: Dict[str, Hypothesis] = torch.load("../hyps_st_15_tmp_3_noglobal.pt",
            #                                              map_location=torch.device('cpu'))

            # saved_hyp_files = [
            #     "../hyps_st_15.pt",
            #     "../hyps_st_15_tmp_2.pt",
            #     "../hyps_st_15_tmp_3.pt",
            #     "../hyps_st_15_tmp_4.pt",
            #     "../hyps_st_15_tmp_5.pt",
            #     "../hyps_st_15_tmp_10.pt",
            #     "../hyps_st_15_tmp_100.pt",
            #     "../hyps_st_15_tmp_2_noglobal.pt",
            #     "../hyps_st_15_tmp_3_noglobal.pt",
            #     "../hyps_st_15_tmp_4_noglobal.pt",
            #     "../hyps_st_15_tmp_5_noglobal.pt",
            #     "../hyps_st_15_tmp_10_noglobal.pt",
            #     "../hyps_st_15_tmp_100_noglobal.pt",
            # ]
            #

            # correct_hyps = 0
            # correct_median = 0
            #
            # def calc_score(history: FloatTensor, tot_score: FloatTensor):
            #     summed_logits = torch.sum(history)
            #     min_logits = torch.min(history)
            #     return min_logits
            #
            # def calc_median(hyp: Hypothesis, fname: str):
            #     if hyp.all_l2r_hyps is None or hyp.all_r2l_hyps is None or (len(hyp.all_l2r_hyps) == 0) or (
            #             len(hyp.all_r2l_hyps) == 0):
            #         return hyp.seq, hyp.history
            #     min_l2r = min((len(hyp.all_l2r_hyps), 2))
            #     min_r2l = min((len(hyp.all_r2l_hyps), 2))
            #     best_l2r_scores, best_l2r_idx = torch.topk(hyp.all_l2r_scores, k=min_l2r)
            #     best_r2l_scores, best_r2l_idx = torch.topk(hyp.all_r2l_scores, k=min_r2l)
            #     bstrs = []
            #     wlist = []
            #     abs_best_l2r = []
            #     abs_best_l2r_history = []
            #     abs_best_r2l = []
            #     abs_best_r2l_history = []
            #     for best, score in enumerate(best_l2r_scores):
            #         if best == 0:
            #             abs_best_l2r = hyp.all_l2r_hyps[best_l2r_idx[best]].tolist()
            #             abs_best_l2r_history = hyp.all_l2r_history[best_l2r_idx[best]]
            #         bstrs.append(bytes(hyp.all_l2r_hyps[best_l2r_idx[best]].tolist()))
            #         wlist.append(1 / (100 * abs(float(score))))
            #     for best, score in enumerate(best_r2l_scores):
            #         if best == 0:
            #             abs_best_r2l = hyp.all_r2l_hyps[best_r2l_idx[best]].tolist()
            #             abs_best_r2l_history = hyp.all_r2l_history[best_r2l_idx[best]]
            #         bstrs.append(bytes(hyp.all_r2l_hyps[best_r2l_idx[best]].tolist()))
            #         wlist.append(1 / (100 * abs(float(score))))
            #     if len(abs_best_l2r) == 0:
            #         abs_best_l2r = abs_best_r2l
            #         abs_best_l2r_history = abs_best_r2l_history
            #     if len(abs_best_r2l) == 0:
            #         abs_best_r2l = abs_best_l2r
            #         abs_best_r2l_history = abs_best_l2r_history
            #
            #     mstr = list(bytearray(median(bstrs, wlist), "utf-8"))
            #     mhistory = hyp.history.copy()
            #     if len(mhistory) != len(mstr):
            #         if len(mhistory) > len(mstr):
            #             mhistory = mhistory[:len(mstr)]
            #         else:
            #             for i in range(abs(len(mhistory) - len(mstr))):
            #                 mhistory.append(hyp.score / 2)
            #     # mstr = hyp.seq
            #     mstr_len = len(mstr)
            #
            #     gt = oracle.get_gt_indices(fname)
            #     gt_len = len(gt)
            #
            #     for i, token in enumerate(abs_best_l2r):
            #         if (i >= gt_len) or (i >= mstr_len) or (gt[i] != token):
            #             break
            #         mstr[i] = token
            #         mhistory[i] = abs_best_l2r_history[i]
            #
            #     i = 0
            #     for r2l_i, token in reversed(list(enumerate(abs_best_r2l))):
            #         if (i >= gt_len) or (i >= mstr_len) or gt[gt_len - i - 1] != token:
            #             break
            #         mstr[mstr_len - 1 - i] = token
            #         mhistory[mstr_len - 1 - i] = abs_best_r2l_history[r2l_i]
            #         i += 1
            #
            #     return mstr, mhistory
            #
            # counters = defaultdict(float)
            # total_conf_correct = 0.0
            # min_correct_conf = float('Inf')
            #
            # def calc_vec(hyp: Hypothesis):
            #     token_vec = np.zeros(vocab.__len__() + 1)
            #     total = len(hyp.all_r2l_hyps) + len(hyp.all_l2r_hyps)
            #     token_vec[vocab.__len__()] = total
            #     for idx, token in enumerate(hyp.seq):
            #         token_vec[token] += hyp.history[idx] * 100 * total
            #
            #     # token_vec = np.zeros(400)
            #     # for idx, token in enumerate(hyp.seq):
            #     #     token_vec[idx] = token
            #     #     token_vec[200 + idx] = hyp.history[idx]
            #
            #     # token_vec = np.zeros(vocab.__len__() * 2)
            #     # total = len(hyp.all_r2l_hyps) + len(hyp.all_l2r_hyps)
            #     # for idx, token in enumerate(hyp.seq):
            #     #     token_vec[token] += hyp.history[idx] * 100 * total
            #     #     token_vec[vocab.__len__() + token] += hyp.best_rev[idx] * 100 * total
            #     return token_vec
            #
            # X_normal = []
            # y_normal = []
            # has_correct = False
            # has_incorrect = False
            #
            # def tuple_factory():
            #     return {
            #         "total": 0,
            #         "correct": 0,
            #         "missed": 0
            #     }
            #
            # def bin_factory():
            #     return defaultdict(tuple_factory)
            #
            # bins = defaultdict(bin_factory)
            # rated_hyps = defaultdict(list)
            # hyps = []
            # levs = []
            #
            # for fname, hyp in all_hyps.items():
            #     lev_dist = oracle.levenshtein_indices(fname, hyp.seq)
            #     y = 0
            #
            #     hyp_len = len(hyp.seq)
            #
            #     if hyp_len > 0:
            #         hyps.append(hyp)
            #         levs.append(lev_dist)
            #     else:
            #         continue
            #
            #     is_correct = hyp_len > 0 and lev_dist == 0
            #
            #     if lev_dist == 0:
            #         bins["oracle"][hyp_len]["total"] += 1
            #         bins["oracle"][hyp_len]["correct"] += 1
            #         y = 1
            #         counters["correct"] += 1
            #         min_conf = min(hyp.history)
            #         total_conf_correct += min_conf
            #         if min_conf < min_correct_conf:
            #             min_correct_conf = min_conf
            #
            #     mpred, mhistory = calc_median(hyp, fname)
            #     mpred_lev_dist = oracle.levenshtein_indices(fname, mpred)
            #     if mpred_lev_dist == 0:
            #         counters["median_correct"] += 1
            #
            #     median_min_score = min(mhistory)
            #     rated_hyps["median"].append(median_min_score)
            #
            #     if len(mhistory) > 0 and median_min_score >= th:
            #         counters["median_conf_passed"] += 1
            #         bins["median"][hyp_len]["total"] += 1
            #         if mpred_lev_dist == 0:
            #             bins["median"][hyp_len]["correct"] += 1
            #             counters["median_conf_passed_correct"] += 1
            #         else:
            #             counters["median_conf_lev_dist"] += mpred_lev_dist
            #     elif is_correct:
            #         bins["median"][hyp_len]["missed"] += 1
            #
            #     bimin_score = score_bimin(hyp)
            #     rated_hyps["bimin"].append(bimin_score)
            #
            #     if len(hyp.history) > 0 and bimin_score >= th:
            #         counters["min_biconf_passed"] += 1
            #         bins["bimin"][hyp_len]["total"] += 1
            #         if lev_dist == 0:
            #             bins["bimin"][hyp_len]["correct"] += 1
            #             counters["min_biconf_rev_score_correct"] += min(hyp.best_rev)
            #             counters["min_biconf_passed_correct"] += 1
            #         else:
            #             counters["min_biconf_rev_score_incorrect"] += min(hyp.best_rev)
            #             counters["min_biconf_lev_dist"] += lev_dist
            #     elif is_correct:
            #         bins["bimin"][hyp_len]["missed"] += 1
            #
            #     min_score = score_min(hyp)
            #     rated_hyps["min"].append(min_score)
            #
            #     if len(hyp.history) > 0 and min_score >= th:
            #         bins["min"][hyp_len]["total"] += 1
            #         if np.random.random() < 0.1:
            #             X_normal.append(calc_vec(hyp))
            #             y_normal.append(y)
            #             if y:
            #                 has_incorrect = True
            #             else:
            #                 has_correct = True
            #         counters["min_conf_passed"] += 1
            #         if lev_dist == 0:
            #             bins["min"][hyp_len]["correct"] += 1
            #             counters["min_conf_passed_correct"] += 1
            #         else:
            #             counters["min_conf_lev_dist"] += lev_dist
            #     elif is_correct:
            #         bins["min"][hyp_len]["missed"] += 1
            #
            #     avg_score = score_avg(hyp)
            #     rated_hyps["avg"].append(avg_score)
            #
            #     if len(hyp.history) > 0 and avg_score >= th:
            #         bins["avg"][hyp_len]["total"] += 1
            #         if lev_dist == 0:
            #             bins["avg"][hyp_len]["correct"] += 1
            #     elif is_correct:
            #         bins["avg"][hyp_len]["missed"] += 1
            #
            # print("Oracle", len(all_hyps), counters["correct"],
            #       f"{counters['correct'] * 100 / len(all_hyps):.2f}",
            #       f'{math.exp(total_conf_correct / counters["correct"])}',
            #       math.exp(min_correct_conf)
            #       )
            # # print(len(all_hyps), counters["median_correct"], f"{zero_safe_division(counters['median_correct'] * 100, len(all_hyps)):.2f}")
            # print("MinConf", counters["min_conf_passed"],
            #       f'{zero_safe_division(counters["min_conf_passed_correct"] * 100, counters["min_conf_passed"]):.2f}',
            #       zero_safe_division(counters["min_conf_lev_dist"],
            #                          (counters["min_conf_passed"] - counters["min_conf_passed_correct"])),
            #       )
            # print("MinBiConf", counters["min_biconf_passed"],
            #       f'{zero_safe_division(counters["min_biconf_passed_correct"] * 100, counters["min_biconf_passed"]):.2f}',
            #       zero_safe_division(counters["min_biconf_lev_dist"],
            #                          (counters["min_biconf_passed"] - counters["min_biconf_passed_correct"])),
            #       "Corr. AVG Min (R): ",
            #       f'{zero_safe_exp(zero_safe_division(counters["min_biconf_rev_score_correct"], counters["min_biconf_passed_correct"])):.6f} '
            #       "Incor. AVG Min (R): ",
            #       f'{zero_safe_exp(zero_safe_division(counters["min_biconf_rev_score_incorrect"], (counters["min_biconf_passed"] - counters["min_biconf_passed_correct"]))):.6f} '
            #       )
            #
            # print("### Conf by Lens ###")
            # for name, len_bins in bins.items():
            #     print(f"## {name} ##".center(20))
            #     for hyp_len, total_correct_dict in len_bins.items():
            #         print(f"{hyp_len} : {total_correct_dict['correct']} "
            #               f"/ {total_correct_dict['total']} ({zero_safe_division(100.0 * total_correct_dict['correct'], total_correct_dict['total']):.2f}%) missed: {total_correct_dict['missed']}")
            #
            # percentages = [0.1, 0.2, 0.3, 0.4, 0.5];
            # print("### Best Percentage ###")
            #
            # for p in percentages:
            #     for name, confs in rated_hyps.items():
            #         indices = np.argsort(np.array(confs))[::-1]
            #         correct, total, total_lev = 0, 0, 0
            #         for i in range(int(math.ceil(len(indices) * p))):
            #             total_lev += levs[indices[i]]
            #             if levs[indices[i]] == 0:
            #                 correct += 1
            #             total += 1
            #         print(f"{name}, best {p * 100:.0f}%: {zero_safe_division(correct * 100, total):.2f} lev: {zero_safe_division(total_lev, total):.3f}")

            # print("MedianMinConf", counters["median_conf_passed"],
            #       f'{zero_safe_division(counters["median_conf_passed_correct"] * 100, counters["median_conf_passed"]):.2f}',
            #       zero_safe_division(counters["median_conf_lev_dist"], (counters["median_conf_passed"] - counters["median_conf_passed_correct"]))
            #       )

            # if has_correct and has_incorrect:
            #     clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            #     clf.fit(X_normal, y_normal)
            #
            #     for fname, hyp in all_hyps.items():
            #         if len(hyp.history) > 0 and min(hyp.history) >= th and clf.predict(calc_vec(hyp).reshape(1, -1)) == 1:
            #             counters["svm_passed"] += 1
            #             lev_dist = oracle.levenshtein_indices(fname, hyp.seq)
            #             if lev_dist == 0:
            #                 counters["svm_passed_correct"] += 1
            #             else:
            #                 counters["svm_lev_dist"] += lev_dist
            #
            #             if min(hyp.best_rev) >= th:
            #                 counters["svm_bi_passed"] += 1
            #                 lev_dist = oracle.levenshtein_indices(fname, hyp.seq)
            #                 if lev_dist == 0:
            #                     counters["svm_bi_passed_correct"] += 1
            #                 else:
            #                     counters["svm_bi_lev_dist"] += lev_dist
            #
            #     print("SVM (MIN)", counters["svm_passed"],
            #           f'{zero_safe_division(counters["svm_passed_correct"] * 100, counters["svm_passed"]):.2f}',
            #           zero_safe_division(counters["svm_lev_dist"], (counters["svm_passed"] - counters["svm_passed_correct"]))
            #           )
            #     print("SVM (BIMIN)", counters["svm_bi_passed"],
            #           f'{zero_safe_division(counters["svm_bi_passed_correct"] * 100, counters["svm_bi_passed"]):.2f}',
            #           zero_safe_division(counters["svm_bi_lev_dist"], (counters["svm_bi_passed"] - counters["svm_bi_passed_correct"]))
            #       )

            # curr_min = 1e-10
            # curr_max = 0.95
            #
            # steps = 100000
            # step_size = (curr_max - curr_min) / steps
            #
            # cov_exp = 2
            #
            # inputs = [(curr_min + step_size * s, all_hyps, oracle) for s in range(steps)]
            #
            # do_calc = False
            #
            # if do_calc:
            #     results = []
            #     with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            #         result_iter = p.imap_unordered(single_step, inputs,
            #                                        chunksize=int(math.ceil(len(inputs) / multiprocessing.cpu_count())))
            #         for x in result_iter:
            #             results.append(x)
            #     torch.save(results, f"test_{use_fn.__name__}.pt")
            # else:
            #     results = torch.load(f"test_{use_fn.__name__}.pt")
            #
            # for corr_exp_ in range(30):
            #     corr_exp = (corr_exp_ + 1) / 2
            #     best = 0.95
            #     best_score = float('-Inf')
            #     best_cov, best_pass, best_corr, best_pct, best_exp = 0, 0, 0, 0, 0
            #     for ((m_pass, m_corr, cov_pct, correct_pct), curr_th) in results:
            #         if ((correct_pct ** corr_exp) * (cov_pct ** cov_exp)) > best_score:
            #             best = curr_th
            #             best_score = ((correct_pct ** corr_exp) * (cov_pct ** cov_exp))
            #             best_cov = cov_pct
            #             best_pass = m_pass
            #             best_corr = m_corr
            #             best_pct = correct_pct
            #             best_exp = 1.0
            #     print(f"{corr_exp}".ljust(3), f"{best:.5f}", best_pass, best_corr, f"corr: {best_pct * 100:.2f}",
            #           f"cov: {best_cov * 100:.2f}")

    # for i in range(steps):
    #     curr = curr_min + step_size * i
    #     for x in range(exp_steps):
    #         exp = exp_min + x * exp_step_size
    #         m_pass, m_corr, cov_pct, correct_pct = calc_min(curr, exp)
    #
    #         # print(curr, cov_pct, correct_pct, (cov_pct * correct_pct))
    #
    #         if (m_corr * (correct_pct ** cov_exp)) > best_score:
    #             best = curr
    #             best_score = (m_corr * (correct_pct ** cov_exp))
    #             best_cov = cov_pct
    #             best_pass = m_pass
    #             best_corr = m_corr
    #             best_pct = correct_pct
    #             best_exp = exp
    # print(best, best_exp, best_score, best_cov, best_pass, best_corr, best_pct)
    # print(calc_min(0.9875, all_hyps, oracle))

    # min = 0.95
    # change = 1

    # print("Hyps", len(all_hyps), "Correct", correct_hyps)
    # print("AVG", "Passed: ", avg_conf_passed, " Correct: ", avg_conf_passed_correct, f"{avg_conf_passed_correct * 100 / avg_conf_passed:.2f}", avg_conf_lev_sum / (avg_conf_passed - avg_conf_passed_correct))
    # print("MIN", "Passed: ", min_conf_passed, " Correct: ", min_conf_passed_correct, f"{min_conf_passed_correct * 100 / min_conf_passed:.2f}", min_conf_lev_sum / (min_conf_passed - min_conf_passed_correct))
    #

    # entries = extract_data_entries(archive, "2014", to_device=device)
    #
    # batch_tuple: List[BatchTuple] = build_batches_from_samples(
    #     entries,
    #     4,
    #     batch_imagesize=(2200 * 250 * 4),
    #     max_imagesize=(2200 * 250),
    #     include_last_only_full=True
    # )[::-1]
    #
    # batch: Batch = collate_fn([batch_tuple[0]]).to(device)

    # batch_tuple_single: List[BatchTuple] = build_batches_from_samples(
    #     entries,
    #     1,
    #     batch_imagesize=(2200 * 250 * 4),
    #     max_imagesize=(2200 * 250),
    #     include_last_only_full=True
    # )[::-1]
    # test: BatchTuple = batch_tuple_single[2]
    # test[0].append(batch_tuple_single[3][0][0])
    # test[1].append(batch_tuple_single[3][1][0])
    # test[2].append(batch_tuple_single[3][2][0])
    # batch_single: Batch = collate_fn([test]).to(device)

    # feature, mask = model.comer_model.encoder(batch_single.imgs, batch_single.mask)
    # corner_vecs = torch.cat((
    #     feature[1, 0, 0, :],
    #     feature[1, 11, 0, :],
    #     feature[1, 0, 70, :],
    #     feature[1, 11, 70, :]
    # ))
    # corner_vecs_m = torch.tensor([
    #     mask[1, 0, 0],
    #     mask[1, 11, 0],
    #     mask[1, 0, 70],
    #     mask[1, 11, 70]
    # ])
    #
    # np.savetxt("test_1_ft.txt", corner_vecs.cpu().numpy())
    # np.savetxt("test_1_mask.txt", corner_vecs_m.cpu().numpy())
    #
    # feature, mask = model.comer_model.encoder(batch.imgs, batch.mask)
    # corner_vecs = torch.cat((
    #     feature[1, 0, 0, :],
    #     feature[1, 11, 0, :],
    #     feature[1, 0, 70, :],
    #     feature[1, 11, 70, :]
    # ))
    # corner_vecs_m = torch.tensor([
    #     mask[1, 0, 0],
    #     mask[1, 11, 0],
    #     mask[1, 0, 70],
    #     mask[1, 11, 70]
    # ])
    #
    # np.savetxt("test_2_ft.txt", corner_vecs.cpu().numpy())
    # np.savetxt("test_2_mask.txt", corner_vecs_m.cpu().numpy())

    # hyps: List[Hypothesis] = model.approximate_joint_search(batch.imgs, batch.mask, use_new=True, debug=False)
    #
    # score_batch = Batch(batch.img_bases, batch.imgs, batch.mask,[hyp.seq for hyp in hyps], 0, 0, 4).to(device)
    #
    # tgt, out = to_bi_tgt_out(score_batch.labels, device)
    # logits = F.log_softmax(
    #     model(score_batch.imgs, score_batch.mask, tgt),
    #     dim=-1
    # )
    #
    # batch_len = len(score_batch)
    #
    # for i, fname in enumerate(batch.img_bases):
    #     print(f"{fname}: (hyp l2r: {hyps[i].was_l2r})")
    #     print(f"{hyps[i].score:.4f}".ljust(5), f"{oracle.confidence_indices(fname, hyps[i].seq):.4f}")
    #     gt = batch.labels[i]
    #     gt_len = len(gt)
    #     pred = hyps[i].seq
    #     pred_len = len(pred)
    #     lines = [[], [], [], [], [], [], [], []]
    #     diff = gt_len - pred_len
    #     for sym in range(gt_len):
    #         str_gt_l2r = vocab.idx2word[gt[sym]]
    #         str_pred_l2r = ""
    #         str_pred_r2l = ""
    #         logit_l2r_str = ""
    #         logit_r2l_str = ""
    #         logit_str = ""
    #         logit_r2l_str_gt_sym = ""
    #
    #         if sym < pred_len:
    #             str_pred_l2r = vocab.idx2word[pred[sym]]
    #             logit_l2r_str = f"{logits[i, sym, pred[sym]]:.1E}"
    #             single_logit = hyps[i].history[sym]
    #             logit_str = f"{single_logit:.1E}"
    #         if sym >= diff:
    #             str_pred_r2l = vocab.idx2word[pred[sym - diff]]
    #             logit_r2l_str = f"{logits[i + batch_len, pred_len - sym + diff - 1, pred[sym - diff]]:.1E}"
    #             logit_r2l_str_gt_sym = f"{logits[i + batch_len, pred_len - sym + diff - 1, gt[sym]]:.1E}"
    #
    #
    #         maxlen = max((len(str_gt_l2r), len(str_pred_l2r),
    #                       len(str_pred_r2l), len(logit_str), len(logit_l2r_str),
    #                       len(logit_r2l_str), len(logit_r2l_str_gt_sym)))
    #         lines[0].append(str_gt_l2r.center(maxlen + 1))
    #         lines[1].append(str_pred_l2r.center(maxlen + 1))
    #         lines[2].append(logit_str.ljust(maxlen + 1))
    #         lines[3].append(logit_l2r_str.ljust(maxlen + 1))
    #
    #         lines[4].append(str_gt_l2r.center(maxlen + 1))
    #         lines[5].append(str_pred_r2l.center(maxlen + 1))
    #         lines[6].append(logit_r2l_str.ljust(maxlen + 1))
    #         lines[7].append(logit_r2l_str_gt_sym.ljust(maxlen + 1))
    #     print("".join(lines[0]))
    #     print("".join(lines[1]))
    #     print("".join(lines[2]))
    #     print("".join(lines[3]))
    #     print()
    #     print("".join(lines[4]))
    #     print("".join(lines[5]))
    #     print("".join(lines[2]))
    #     print("".join(lines[6]))
    #     print("".join(lines[7]))
    #     print(f"len gt: {len(batch.labels[i])} vs. pred: {len(hyps[i].seq)}")

    # trans_list = [ToPILImage(), RandAugment(3), ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI)]
    # transform = tr.Compose(trans_list)
    #
    # file_names, images, labels, is_labled, src_idx = batch_tuple[0]
    #
    # transformed_ims = [transform(im) for im in images]
    #
    # Image.show(ToPILImage()(transformed_ims[0]))
    # Image.show(ToPILImage()(transformed_ims[1]))
    # Image.show(ToPILImage()(transformed_ims[2]))
    # Image.show(ToPILImage()(transformed_ims[3]))

    # start_time = time.time()
    # for idx, batch_tup in enumerate(batch_tuple):
    #     if idx > 0:
    #         break
    #     batch: Batch = collate_fn([batch_tup]).to(device)
    #     hyps = model.approximate_joint_search(batch.imgs, batch.mask, use_new=True, debug=False)
    #     for i, hyp in enumerate(hyps):
    #         print(batch.img_bases[i], vocab.indices2words(hyp.seq))
    # hyps = model.approximate_joint_search(batch.imgs, batch.mask, use_new=False)
    # hyps_new = model.approximate_joint_search(batch.imgs, batch.mask, use_new=True)
    # for i, hyp_old in enumerate(hyps):
    #     if hyp_old.seq != hyps_new[i].seq:
    #         print("OLD:")
    #         model.approximate_joint_search(batch.imgs, batch.mask, use_new=False, debug=True)
    #         print("")
    #         print("")
    #         print("NEW:")
    #         model.approximate_joint_search(batch.imgs, batch.mask, use_new=True, debug=True)
    #         print("mismatch", batch.img_bases[0], idx)
    #         print("old: ", vocab.indices2words(hyp_old.seq))
    #         print("new: ", vocab.indices2words(hyps_new[i].seq))
    #         exit(1)
    # prof.export_stacks("profiler_stacks_gpu.txt", "self_cuda_time_total")
    # prof.export_stacks("profiler_stacks_cpu.txt", "self_cpu_time_total")
    # torch.save(prof.key_averages().table(sort_by="self_cpu_time_total"), "profiler_table.txt")
    # prof.export_chrome_trace("profiler_chrome.json")
    # print("total: ", time.time() - start_time)

    # n = 2
    # print(f"benching normal {n} times")
    # start_time = time.time()
    # for _ in range(n):
    #     full(batch, model, shapes, use_new=False)
    # print("total time: ", time.time() - start_time)
    #
    # print(f"benching new {n} times")
    # start_time = time.time()
    # for _ in range(n):
    #     full(batch, model, shapes, use_new=True)
    # print("total time: ", time.time() - start_time)


def zero_safe_division(n, d):
    return n / d if d else 0


def zero_safe_exp(e):
    return math.exp(e) if e else e


def full(batch: Batch, model, shapes, use_new: bool = False):
    return model.approximate_joint_search(batch.imgs, batch.mask, use_new=use_new)
    # print(hyps[0].score, len(hyps[0].seq), vocab.indices2words(hyps[0].seq))


def gauss_sum(n):
    return n * (n + 1) / 2


def masked_mean_var(logits):
    inputs = np.array(logits)
    idx = np.argmin(inputs)

    masked = np.ma.array(inputs, mask=False)
    masked.mask[idx] = True

    m = masked.mean()
    power = np.dot(masked, masked) / masked.size
    var = power - m ** 2
    return m, var, idx, masked


def get_best_l2r_data(hyp: Hypothesis):
    if hyp.was_l2r:
        return hyp.seq, hyp.history, hyp.best_rev
    else:
        best_idx = np.argmax(np.array(hyp.all_l2r_scores))
        return hyp.all_l2r_hyps[best_idx], hyp.all_l2r_history[best_idx], hyp.all_l2r_rev_scores[best_idx]


def get_best_r2l_data(hyp: Hypothesis):
    if not hyp.was_l2r:
        return hyp.seq, hyp.history, hyp.best_rev
    else:
        best_idx = np.argmax(np.array(hyp.all_r2l_scores))
        return hyp.all_r2l_hyps[best_idx], hyp.all_r2l_history[best_idx], hyp.all_r2l_rev_scores[best_idx]


def partial_label(hyp: Hypothesis,
                  std_fac: float = 1.0,
                  fname: Union[str, None] = None,
                  oracle: Union[Oracle, None] = None,
                  partial_mode=0,
                  threshold: float = float('-Inf')) -> Tuple[bool, List[int], Union[List[int], None]]:
    if oracle is not None and fname is not None:
        label = oracle.get_gt_indices(fname)
        if label == hyp.seq:
            return False, hyp.seq, None

        hyp_len, label_len = len(hyp.seq), len(label)
        min_len = min((hyp_len, label_len))

        if label[:min_len] == hyp.seq[:min_len]:
            l2r_seq = hyp.seq[:min_len]
        else:
            first_mismatch_l2r = next(compress(count(), map(ne, label[:min_len], hyp.seq[:min_len])))
            l2r_seq = hyp.seq[:first_mismatch_l2r]

        if label[-min_len:] == hyp.seq[-min_len:]:
            r2l_seq = hyp.seq[-min_len:]
        else:
            first_mismatch_r2l = next(
                compress(count(), map(ne, reversed(label[-min_len:]), reversed(hyp.seq[-min_len:]))))
            r2l_seq = hyp.seq[hyp_len - first_mismatch_r2l:]
        return True, l2r_seq, r2l_seq
    if len(hyp.seq) < 2:
        return False, hyp.seq, None
    # hyp_len = len(hyp.seq)
    # logits = np.array(hyp.history)
    # l2r_means = np.cumsum() / (np.arange(1, hyp_len + 1))
    #
    # power = np.dot(logits, logits) / logits.size
    # var = power - l2r_means[-1] ** 2
    # std = np.sqrt(var)
    # for i in range(hyp_len):
    #     if

    if partial_mode == 0:
        # find the smallest logit, if it is farther than X*std from the mean of the rest, use left/right side of the idx
        # as partial hyps
        avgs = np.array([(hyp.history[i] + hyp.best_rev[i]) / 2 for i in range(len(hyp.seq))])
        idx = np.argmin(avgs)
        masked_avgs = np.ma.array(avgs, mask=False)
        masked_avgs.mask[idx] = True

        m = masked_avgs.mean()
        power = np.dot(masked_avgs, masked_avgs) / masked_avgs.size
        var = power - m ** 2
        std = np.sqrt(var)
        min_dev = m - avgs[idx]
        if min_dev >= (std * std_fac):
            # mask it and use l2r / r2l from there
            return True, hyp.seq[:idx], hyp.seq[idx + 1:]
        return False, hyp.seq, None
    elif partial_mode == 1:
        # Check the worst 2 logits, if they are farther than X*std from the mean of the remaining logits
        # use lower/upper indices to cap l2r/r2l hyps
        np_hist = np.array(hyp.history)
        np_rev = np.array(hyp.best_rev)
        avgs = (np_hist + np_rev) / 2
        m, var, idx, masked = masked_mean_var(avgs)
        std = np.sqrt(var)

        min_dev = m - avgs[idx]

        if min_dev >= (std * std_fac):
            if len(hyp.seq) < 3:
                return True, hyp.seq[:idx], hyp.seq[idx + 1:]
            second_smallest = masked.argmin(fill_value=1.0)
            masked.mask[second_smallest] = True

            m = masked.mean()
            power = np.dot(masked, masked) / masked.size
            var = power - m ** 2
            std = np.sqrt(var)

            second_min_dev = m - avgs[idx]
            if second_min_dev >= (std * std_fac):
                bottom_idx, top_idx = min((idx, second_smallest)), max((idx, second_smallest))
                return True, hyp.seq[:bottom_idx], hyp.seq[top_idx + 1:]
            # std = np.sqrt(var)

            # mask it and use l2r / r2l from there
            return True, hyp.seq[:idx], hyp.seq[idx + 1:]
        return False, hyp.seq, None
    elif partial_mode == 2:
        # use the best l2r hyp for the l2r-partial
        # and the best r2l hyp for the r2l-partial
        # performs worse, since not the "best" hyp is taken each time
        l2r_hyp, l2r_hist, l2r_rev = get_best_l2r_data(hyp)
        l2r = l2r_hyp

        if len(l2r_hyp) > 1:
            m, var, idx, _ = masked_mean_var(l2r_hist)
            std = np.sqrt(var)
            min_dev = m - l2r_hist[idx]
            if min_dev >= (std * std_fac):
                l2r = l2r_hyp[:idx]

        r2l_hyp, r2l_hist, r2l_rev = get_best_r2l_data(hyp)
        r2l = r2l_hyp

        if len(r2l_hyp) > 1:
            m, var, idx, _ = masked_mean_var(r2l_hist)
            std = np.sqrt(var)
            min_dev = m - r2l_hist[idx]
            if min_dev >= (std * std_fac):
                r2l = r2l_hyp[idx + 1:]

        return True, l2r, r2l
    elif partial_mode == 3:
        # use the best hyp, but only use the l2r logit history to partial the l2r hyp
        # and the r2l logit history to partial the r2l hyp
        l2r = hyp.seq
        r2l = hyp.seq

        if hyp.was_l2r:
            m, var, idx, _ = masked_mean_var(hyp.history)
            std = np.sqrt(var)
            min_dev = m - hyp.history[idx]
            if min_dev >= (std * std_fac):
                l2r = hyp.seq[:idx]

            m, var, idx, _ = masked_mean_var(hyp.best_rev)
            std = np.sqrt(var)
            min_dev = m - hyp.best_rev[idx]
            if min_dev >= (std * std_fac):
                r2l = hyp.seq[idx + 1:]
        else:
            m, var, idx, _ = masked_mean_var(hyp.best_rev)
            std = np.sqrt(var)
            min_dev = m - hyp.best_rev[idx]
            if min_dev >= (std * std_fac):
                l2r = hyp.seq[:idx]

            m, var, idx, _ = masked_mean_var(hyp.history)
            std = np.sqrt(var)
            min_dev = m - hyp.history[idx]
            if min_dev >= (std * std_fac):
                r2l = hyp.seq[idx + 1:]

        return True, l2r, r2l
    elif partial_mode == 4:
        # use a weighted average between l2r/r2l logits when searching for the minimum
        # and the r2l logit history to partial the r2l hyp
        l2r = hyp.seq
        r2l = hyp.seq
        np_hist = np.array(hyp.history)
        np_rev = np.array(hyp.best_rev)

        w = 3

        if hyp.was_l2r:
            weighted_avgs = (w * np_hist + np_rev) / (w + 1)
            m, var, idx, _ = masked_mean_var(weighted_avgs)
            std = np.sqrt(var)
            min_dev = m - weighted_avgs[idx]
            if min_dev >= (std * std_fac):
                l2r = hyp.seq[:idx]

            weighted_avgs = (np_hist + w * np_rev) / (w + 1)
            m, var, idx, _ = masked_mean_var(weighted_avgs)
            std = np.sqrt(var)
            min_dev = m - weighted_avgs[idx]
            if min_dev >= (std * std_fac):
                r2l = hyp.seq[idx + 1:]
        else:
            weighted_avgs = (np_hist + w * np_rev) / (w + 1)
            m, var, idx, _ = masked_mean_var(weighted_avgs)
            std = np.sqrt(var)
            min_dev = m - weighted_avgs[idx]
            if min_dev >= (std * std_fac):
                l2r = hyp.seq[:idx]

            weighted_avgs = (w * np_hist + np_rev) / (w + 1)
            m, var, idx, _ = masked_mean_var(weighted_avgs)
            std = np.sqrt(var)
            min_dev = m - weighted_avgs[idx]
            if min_dev >= (std * std_fac):
                r2l = hyp.seq[idx + 1:]

        return True, l2r, r2l
    elif partial_mode == 5:
        # find the smallest logit, if it is farther than X*std from the mean of the rest, use left/right side of the idx
        # as partial hyps. Or if the remaining sequence is better than the usual threshold.
        avgs = np.array([(hyp.history[i] + hyp.best_rev[i]) / 2 for i in range(len(hyp.seq))])
        idx = np.argmin(avgs)
        masked_avgs = np.ma.array(avgs, mask=False)
        masked_avgs.mask[idx] = True

        m = masked_avgs.mean()
        power = np.dot(masked_avgs, masked_avgs) / masked_avgs.size
        std = np.sqrt(power - m ** 2)
        min_dev = m - avgs[idx]
        if m >= threshold or (min_dev >= (std * std_fac)):
            # mask it and use l2r / r2l from there
            return True, hyp.seq[:idx], hyp.seq[idx + 1:]
        return False, [], None
    elif partial_mode == 6:
        # find the smallest logit, remaining mean passes the threshold, use left/right side of the idx
        # as partial hyps
        avgs = np.array([(hyp.history[i] + hyp.best_rev[i]) / 2 for i in range(len(hyp.seq))])
        idx = np.argmin(avgs)
        masked_avgs = np.ma.array(avgs, mask=False)
        masked_avgs.mask[idx] = True

        m = masked_avgs.mean()
        if m >= threshold:
            # mask it and use l2r / r2l from there
            return True, hyp.seq[:idx], hyp.seq[idx + 1:]
        return False, [], None
    elif partial_mode == 7:
        # find the smallest logit, use left/right side of the idx as partial hyps.
        avgs = np.array([(hyp.history[i] + hyp.best_rev[i]) / 2 for i in range(len(hyp.seq))])
        idx = np.argmin(avgs)
        return True, hyp.seq[:idx], hyp.seq[idx + 1:]


def eval_sorting_score(oracle: Oracle, partial: bool = False, partial_std_fac: float = 1.0, use_oracle: bool = False):
    # Loads multiple hyp collections from different checkpoints and evaluates multiple conf-measures based
    # on the sorting of the confidence scores
    hyp_files = [
        "../hyps_s_35_new_original_1.pt",
        "../hyps_s_35_new_original_ts_both.pt",
        "../hyps_s_35_new_original_ts_ce.pt",
        "../hyps_s_35_new_original_ts_ece.pt",
        "../hyps_s_35_new_t0_02_opt.pt",
        "../hyps_s_35_new_t0_04_opt.pt",
        "../hyps_s_35_new_t0_05_opt.pt",
        "../hyps_s_35_new_t0_075_opt.pt",
        "../hyps_s_35_new_t0_1_opt.pt",
        "../hyps_s_35_new_t0_5_opt.pt",

    ]
    all_hyps: Dict[str, Dict[str, Hypothesis]] = {}
    for hyp_file in hyp_files:
        all_hyps[hyp_file] = torch.load(hyp_file, map_location=torch.device('cpu'))
    for name, sfn in [("ORI", score_ori), ("AVG", score_avg), ("REV_AVG", score_rev_avg),
                      ("BI_AVG", score_bi_avg), ("BIMIN", score_bimin), ("MULT", score_sum), ("BIMULT", score_bisum)]:
        print(f"{name}".ljust(14), end="")
        for hyps in all_hyps.values():
            print(f"{sorting_score(hyps, sfn, oracle, partial, partial_std_fac, use_oracle):.2f}".ljust(16), end="")
        print()


def sorting_score(hyps, scoring_fn, oracle, partial: bool = False,
                  partial_std_fac: float = 1.0, use_oracle: bool = False,
                  partial_threshold: float = float('Inf')
                  ):
    splitted_fnames = []
    splitted_hyps: List[Tuple[bool, List[int], Union[List[int], None]]] = []
    splitted_scores = []

    for (fname, hyp) in hyps.items():
        splitted_fnames.append(fname)
        score = scoring_fn(hyp)
        if partial and score < partial_threshold:
            splitted_hyps.append(
                partial_label(hyp, partial_std_fac, fname if use_oracle else None, oracle if use_oracle else None))
        else:
            splitted_hyps.append((False, hyp.seq, None))
        splitted_scores.append(score)

    splitted_scores = np.array(splitted_scores)
    splitted_scores_sorted = np.argsort(splitted_scores)[::-1]

    partial_bidir = 2 if partial else 1
    total_hyps = len(splitted_scores_sorted)

    wrong_idx_sum = 0
    wrong_hyps = 0

    skipped_partials = 0
    total_partials = 0

    for best_i, idx in enumerate(splitted_scores_sorted):
        hyp: Tuple[bool, List[int], Union[List[int], None]] = splitted_hyps[idx]
        fname: str = splitted_fnames[idx]
        if partial and hyp[0]:
            total_partials += 1
            l2r_len, r2l_len = len(hyp[1]), len(hyp[2])
            label = oracle.get_gt_indices(fname)

            if l2r_len:
                if label[:l2r_len] != hyp[1]:
                    wrong_idx_sum += best_i
                    wrong_hyps += 1
            else:
                skipped_partials += 1

            if r2l_len:
                if label[-r2l_len:] != hyp[2]:
                    wrong_idx_sum += best_i
                    wrong_hyps += 1
            else:
                skipped_partials += 1
        elif oracle.get_gt_indices(fname) != hyp[1]:
            wrong_idx_sum += best_i * partial_bidir
            wrong_hyps += partial_bidir
    total_hyps = (total_hyps * partial_bidir) - skipped_partials

    correct_hyps = total_hyps - wrong_hyps

    bc_score = gauss_sum(total_hyps / partial_bidir) * partial_bidir - gauss_sum(
        correct_hyps / partial_bidir) * partial_bidir

    wc_score = gauss_sum(wrong_hyps / partial_bidir) * partial_bidir
    return zero_safe_division(bc_score - wrong_idx_sum, bc_score - wc_score) * 100


def average_precision(hyps, scoring_fn, oracle,
                      partial_std_fac: Union[float, None] = None, use_oracle: bool = False,
                      partial_mode=0,
                      calc_tp_as_all_correct: bool = True,
                      partial_threshold: float = 1.0,
                      min_threshold: float = float('-Inf')
                      ):
    do_partial = partial_std_fac is not None
    partial_bidir = 2 if do_partial else 1

    splitted_fnames = []
    splitted_hyps: List[Tuple[bool, List[int], Union[List[int], None]]] = []
    splitted_scores = []

    for (fname, hyp) in hyps.items():
        splitted_fnames.append(fname)
        score = scoring_fn(hyp)
        splitted_scores.append(score)

    splitted_scores = np.array(splitted_scores)
    splitted_scores_sorted = np.argsort(splitted_scores)[::-1]
    # threshold = splitted_scores[
    #     splitted_scores_sorted[
    #         int(math.floor((len(splitted_scores) - 1) * partial_threshold))
    #     ]
    # ]
    threshold = partial_threshold

    for (fname, hyp) in hyps.items():
        score = scoring_fn(hyp)
        if do_partial and score < threshold:
            exp_score = np.exp(score)
            splitted_hyps.append(
                partial_label(
                    hyp, partial_std_fac,
                    fname if use_oracle else None,
                    oracle if use_oracle else None,
                    partial_mode,
                    threshold
                )
            )
        else:
            splitted_hyps.append((False, hyp.seq, None))

    if calc_tp_as_all_correct:
        cumulative_tp = 0
        for best_i, idx in enumerate(splitted_scores_sorted):
            if splitted_scores[idx] < min_threshold:
                continue
            hyp: Tuple[bool, List[int], Union[List[int], None]] = splitted_hyps[idx]
            fname: str = splitted_fnames[idx]
            if do_partial and hyp[0] and splitted_scores[idx] < threshold:
                l2r_len, r2l_len = len(hyp[1]), len(hyp[2])
                label = oracle.get_gt_indices(fname)
                if l2r_len:
                    if label[:l2r_len] == hyp[1]:
                        cumulative_tp += 1
                if r2l_len:
                    if label[-r2l_len:] == hyp[2]:
                        cumulative_tp += 1
            elif oracle.get_gt_indices(fname) == hyp[1]:
                cumulative_tp += partial_bidir
    else:
        cumulative_tp = partial_bidir * len(splitted_scores_sorted)
    total_tp, cumulative_tp = cumulative_tp, 0
    cumulative_fp = 0
    precisions = []
    recalls = []

    skipped_tokens = 0
    total_tokens = 0

    for best_i, idx in enumerate(splitted_scores_sorted):
        hyp: Tuple[bool, List[int], Union[List[int], None]] = splitted_hyps[idx]
        fname: str = splitted_fnames[idx]
        ori_hyp = hyps[fname].seq
        if splitted_scores[idx] < min_threshold:
            continue
        if do_partial and hyp[0] and splitted_scores[idx] < threshold:
            l2r_len, r2l_len = len(hyp[1]), len(hyp[2])
            label = oracle.get_gt_indices(fname)

            if l2r_len:
                total_tokens += len(ori_hyp)
                skipped_tokens += len(ori_hyp) - l2r_len
                if label[:l2r_len] == hyp[1]:
                    cumulative_tp += 1
                else:
                    cumulative_fp += 1

            if r2l_len:
                total_tokens += len(ori_hyp)
                skipped_tokens += len(ori_hyp) - r2l_len
                if label[-r2l_len:] == hyp[2]:
                    cumulative_tp += 1
                else:
                    cumulative_fp += 1
        elif oracle.get_gt_indices(fname) == hyp[1]:
            cumulative_tp += partial_bidir
            total_tokens += 2 * len(ori_hyp)
        else:
            cumulative_fp += partial_bidir
            total_tokens += 2 * len(ori_hyp)

        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall = cumulative_tp / total_tp

        precisions.append(precision)
        recalls.append(recall)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    return precisions, recalls, metrics.auc(recalls, precisions), skipped_tokens, total_tokens


def rate_threshold(hyps: Dict[str, Hypothesis],
                   oracle: Oracle,
                   threshold: float,
                   scoring_fn: Callable[[Hypothesis], float]
                   ):
    # evaluates a single conf-measures on a batch of hypothesis, based on a given threshold,
    # the error rate on passed hyps, average levenshtein error
    passed = 0
    incorrect = 0
    incorrect_lev_dist_sum = 0
    incorrect_with_wrong_len = 0
    for fname, hyp in hyps.items():
        conf = scoring_fn(hyp)
        if conf >= threshold:
            passed += 1

            lev_dist = oracle.levenshtein_indices(fname, hyp.seq)
            is_correct = lev_dist == 0
            correct_len = len(hyp.seq) == len(oracle.get_gt_indices(fname))

            if not is_correct:
                incorrect += 1
                incorrect_lev_dist_sum += lev_dist
                if not correct_len:
                    incorrect_with_wrong_len += 1
    return passed, incorrect, incorrect_with_wrong_len, incorrect_lev_dist_sum


def eval_confs_by_thresholds(hyps: Dict[str, Hypothesis],
                             oracle: Oracle, ):
    # evaluates a multiple conf-measures on a batch of hypothesis, based on multiple thresholds,
    # the error rate on passed hyps, average levenshtein error
    # Print the metrics according to this header:
    # Passed	Err (%)		Err-Len (%)		    Levenshtein
    #                                       Sum	   Ø-Pass Ø-Err
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.925, 0.95, 0.96, 0.975]:
        print(f"{thresh}".ljust(5), end="")
        for sfn in [score_ori, score_avg, score_bimin]:
            passed, incorrect, incorrect_with_wrong_len, incorrect_lev_dist_sum = rate_threshold(hyps, oracle,
                                                                                                 math.log(thresh), sfn)
            print(f" {passed}".ljust(6) +
                  f" {incorrect} ({incorrect * 100 / passed:.1f})".ljust(13) +
                  f" {incorrect_with_wrong_len} ({zero_safe_division(incorrect_with_wrong_len * 100, incorrect):.1f}) ".ljust(
                      13) +
                  f" {incorrect_lev_dist_sum} {incorrect_lev_dist_sum / passed:.2f} ".ljust(11) +
                  f" {zero_safe_division(incorrect_lev_dist_sum, incorrect):.2f}  ".ljust(8), end='')
    print()


def confidence_measure_ap_ece_table(hyps_from_cps: List[Dict[str, Hypothesis]], oracle: Oracle):
    ece = ECELoss()

    def hyp_to_triplet_with_scoring(sfn: Callable[[Hypothesis], float], oracle: Oracle):
        def hyp_to_triplet(tuple: [str, Hypothesis]):
            seq_len = len(tuple[1].seq)
            return np.exp(sfn(tuple[1])) if seq_len else 0.0, tuple[1].seq, oracle.get_gt_indices(tuple[0])

        return hyp_to_triplet

    results = defaultdict(list)
    scoring_fns = [
        ("$\\text{ORI}$", score_ori),
        ("$\\text{AVG}$", score_avg),
        ("$\\text{BIAVG}$", score_bi_avg),
        ("$\\text{MIN}$", score_min),
        ("$\\text{BIMIN}$", score_bimin),
        ("$\\text{MULT}$", score_sum),
        ("$\\text{BIMULT}$", score_bisum)
    ]

    for hyps in hyps_from_cps:
        for name, sfn in scoring_fns:
            precisions, recalls, auc, skips, total = average_precision(
                hyps,
                sfn,
                oracle,
                None,
                calc_tp_as_all_correct=False
            )
            to_triplet = hyp_to_triplet_with_scoring(sfn, oracle)
            ece_score, acc = ece.ece_for_predictions(map(to_triplet, hyps.items()))
            results[name].append(ece_score * 100)
            results[name].append(auc * 100)
    stacked = np.vstack(list(results.values()))
    maxes = np.max(stacked, axis=0)
    mins = np.min(stacked, axis=0)

    for name, sfn in scoring_fns:
        conf_results = np.array(results[name])
        row_string = []
        for i, value in enumerate(conf_results):
            compare_val = mins
            if i % 2:
                compare_val = maxes
            if value == compare_val[i]:
                row_string.append(f"$\\mathbf{{{value:.1f}}}$")
            else:
                row_string.append(f"${value:.1f}$")

        print(f"{name} & {' & '.join(row_string)}  \\\\ \\hline")


def generate_hyps(checkpoints: List[Tuple[str, Any, str]],
                  datasets_with_suffix: List[Tuple[str, List[BatchTuple]]],
                  device: torch.device,
                  temps: List[Union[float, None]] = None,
                  output_root: str = "./"
                  ):
    if temps is None:
        temps = [None]
    with torch.inference_mode():
        for (cp, model_class, name) in checkpoints:
            model = None
            torch.cuda.empty_cache()
            all_hyps = {}

            print(f"Loading {cp}...")
            model: model_class \
                = model_class.load_from_checkpoint(
                cp
            )
            model = model.eval().to(device)
            model.share_memory()
            print("model loaded")

            for (save_ds_suffix, set) in datasets_with_suffix:
                for temp in temps:
                    save_path = f"{output_root}{name}{f'_{temp}' if temp is not None else ''}{save_ds_suffix}.pt"
                    exists = os.path.exists(save_path)
                    if not exists:
                        ten_pct_steps = np.floor(np.linspace(0, len(set), 10, endpoint=False))
                        print(f"LN {name}{save_ds_suffix}, progress: ", end="", flush=True)
                        progress = 0
                        all_hyps = {}
                        for i, batch_raw in enumerate(set):
                            if i in ten_pct_steps:
                                print(progress, end="", flush=True)
                                progress += 1
                            batch = collate_fn([batch_raw]).to(device=device)
                            hyps = model.approximate_joint_search(
                                batch.imgs, batch.mask, use_new=True, debug=False, save_logits=False, temperature=temp,
                                global_pruning='none'
                            )
                            for i, hyp in enumerate(hyps):
                                all_hyps[batch.img_bases[i]] = hyp
                        torch.save(all_hyps, save_path)
                        print(" saved")


if __name__ == '__main__':
    CLI(main)

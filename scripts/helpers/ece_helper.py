import math
import os.path
from itertools import compress, count
from operator import ne
from os import path
from typing import List, Tuple, Union, Callable
from zipfile import ZipFile

import numpy as np
import torch
from jsonargparse import CLI
import matplotlib as mpl
from pytorch_lightning import seed_everything

from comer.datamodules import Oracle
from comer.datamodules.crohme import extract_data_entries, Batch
from comer.utils import ECELoss
from comer.utils.conf_measures import CONF_MEASURES
import sklearn.metrics as metrics

from comer.utils.utils import Hypothesis

OUT_DIR = ""


# \addplot+[ybar] plot coordinates {(tool1,0) (tool2,2)
# 					(tool3,2) (tool4,3) (tool5,0) (tool6,2) (tool7,0)};

# symbolic x coords={tool1, tool2, tool3, tool4,
# 		tool5, tool6, tool7},
def main(paths: List[str], ece: bool = True, ylabel: bool = True, conf: List[str]=None, height: float = 5.0, hyps: bool = False, combine: bool = False):
    if conf is None:
        conf = ['ori']
    device = torch.device("cpu")

    w = (1 / 15)
    # for i in range(16):
    #     red, green, blue =mpl.colormaps["Reds"](((i * w) / 2) + 0.25),\
    #         mpl.colormaps["Greens"](0.8 - ((i * w) / 3)), \
    #         mpl.colormaps["Blues"](((i * w) / 2) + 0.5)
    #     print(f"\\definecolor{{ececrolor_r{i}}}{{rgb}}{{{red[0]}, {red[1]}, {red[2]}}}")
    #     print(f"\\definecolor{{ececrolor_g{i}}}{{rgb}}{{{green[0]}, {green[1]}, {green[2]}}}")
    #     print(f"\\definecolor{{ececrolor_b{i}}}{{rgb}}{{{blue[0]}, {blue[1]}, {blue[2]}}}")
    for name in paths:
        data = torch.load(name, map_location=device)
        if ece:
            if hyps:
                oracle = get_oracle()
                def hyp_to_triplet_with_scoring(sfn: Callable[[Hypothesis], float], oracle: Oracle):
                    def hyp_to_triplet(tuple: [str, Hypothesis]):
                        seq_len = len(tuple[1].seq)
                        return np.exp(sfn(tuple[1])) if seq_len else 0.0, tuple[1].seq, oracle.get_gt_indices(tuple[0])
                    return hyp_to_triplet
                data = [
                    [
                        ECELoss().get_plot_bins(map(hyp_to_triplet_with_scoring(CONF_MEASURES[c], oracle), data.items())) for c in conf
                    ]
                ]
            for exp_idx, (data_per_scoring_fn) in enumerate(data):
                for conf_idx, (accs, confs, samples, (ece_score, tot_acc)) in enumerate(data_per_scoring_fn):
                    coords_accs = []
                    coords_overconf = []
                    coords_underconf = []
                    # x_coords = []
                    for i, val in enumerate(accs):
                        # x_coords.append(f"")


                        mid = ((2 * i + 1) / (2 * len(accs)))
                        if val > mid:
                            coords_accs.append(f"({i / len(accs)},{mid})")
                            coords_underconf.append(f"({i / len(accs)},{val - mid})")
                            coords_overconf.append(f"({i / len(accs)},0.0)")
                        else:
                            coords_accs.append(f"({i / len(accs)},{val})")
                            coords_underconf.append(f"({i / len(accs)},0.0)")
                            coords_overconf.append(f"({i / len(accs)},{mid - val})")
                    xtick = ", ".join([f"{(i*2 + 1)/15:.2f}" for i in range(8)])
                    ylabel_text = "\\exprate{0}"
                    xlabel_text = "Konfidenz"
                    y_tickslabels = "\n                        yticklabels=false,"
                    lines = [
                        f"""\\begin{{tikzpicture}}
                    \\begin{{axis}}[
                        ylabel={{{ylabel_text if ylabel else ''}}},{y_tickslabels if not ylabel else ''}
                        xlabel={{{xlabel_text}}},
                        xmin=0, xmax=1,
                        ymin=0, ymax=1,
                        width=\\textwidth+0.2cm,
                        ylabel shift=-0.2cm,
                        xlabel shift=-0.2cm,
                        height={height}cm,
                        xtick={{{xtick}}},
                        ybar=0pt,
                        ybar stacked,
                        xticklabel style={{
                            /pgf/number format/fixed,
                            /pgf/number format/precision=2,
                            anchor=center, align=center, below=0mm
                        }},
                        ]
                        \\node[anchor=north west] at (rel axis cs: 0.02,0.9) {{$\\text{{ECE}}={ece_score*100:.2f}$}};
                        """
                    ]
                    if ece:
                        for i in range(15):
                            lines.append(f"\\node[label={{[label distance=0.1cm,text depth=-1.0ex,rotate=90]right:{samples[i]}}}] at ({(i / len(accs)) + 0.01},0.0) {{}};")

                        for i, (acc, over, under) in enumerate(zip(coords_accs, coords_overconf, coords_underconf)):
                            to_print = [f"({i / len(accs)},0.0)"] * 15
                            to_print[i] = acc
                            outline_idx = min((i + 5, 15))
                            lines.append(
                                f"\\addplot+[ybar, bar width=1/15, bar shift=1/30, color=ececrolor_b{i}, draw=ececrolor_b{outline_idx}] plot coordinates {{{' '.join(to_print)}}};")
                            to_print[i] = over
                            lines.append(
                                f"\\addplot+[ybar, bar width=1/15, bar shift=1/30, color=ececrolor_r{i}, draw=ececrolor_r{outline_idx}] plot coordinates {{{' '.join(to_print)}}};")
                            to_print[i] = under
                            lines.append(
                                f"\\addplot+[ybar, bar width=1/15, bar shift=1/30, color=ececrolor_g{i}, draw=ececrolor_g{outline_idx}] plot coordinates {{{' '.join(to_print)}}};")



                    lines.append(f"""\\end{{axis}}
                \\end{{tikzpicture}}%""")

                    with open(os.path.join(OUT_DIR, f"{os.path.basename(name)}_ece_{exp_idx}_{'_'.join(conf)}.tikz"), "w") as f:
                        f.writelines(lines)
        else:
            ylabel_text = "Precision"
            xlabel_text = "Recall"
            threshold, min_threshold = 0.0, 0.0

            oracle = get_oracle()
            threshold = (float('-Inf') if threshold == 0.0 else np.log(threshold))
            min_threshold = float('-Inf') if min_threshold == 0.0 else np.log(min_threshold)

            colors = ['default_r', 'default_g', 'default_b']

            lines = [
                f"""\\begin{{tikzpicture}}
                    \\begin{{axis}}[
                        ylabel={{{ylabel_text if ylabel else ""}}},
                        xlabel={{{xlabel_text}}},
                        xmin=0, xmax=1,
                        ymax=1,
                        width=\\textwidth+0.1cm,
                        height={height}cm,
                        xtick={{0, 0.2, ..., 1.0}},
                        xtick align=outside,
                        ylabel shift=-0.2cm,
                        xlabel shift=-0.2cm,
                        xticklabel style={{
                            /pgf/number format/fixed,
                            /pgf/number format/precision=2,
                            anchor=center, align=center, below=0mm
                        }},
                        ]
                        """
            ]

            for i, c in enumerate(conf):
                assert c in CONF_MEASURES
                precisions, recalls, auc, skips, total = average_precision(
                    data,
                    CONF_MEASURES[c],
                    oracle,
                    None,
                    calc_tp_as_all_correct=True,
                    partial_mode=0,
                    partial_threshold=threshold,
                    min_threshold=min_threshold
                )

                if len(conf) == 1:
                    lines.append(f"\\node[anchor=north east] at (rel axis cs: 0.98,0.9) {{$\\text{{AP}}={auc*100:.2f}$}};")
                else:
                    pass
                    # add legend
                last_recall = 0.0
                filtered_recalls = []
                filtered_precisions = []
                for (recall, prec) in zip(recalls, precisions):
                    if recall == last_recall:
                        continue
                    last_recall = recall
                    filtered_recalls.append(recall)
                    filtered_precisions.append(prec)

                zipped_coords = []
                for (recall, prec) in zip(filtered_recalls, filtered_precisions):
                    zipped_coords.append(f"{recall, prec}")
                lines.append(f"""\\addplot[
                            color={colors[i % len(colors)]},
                            line width=0.35mm,
                            fill=none
                            ]
                            coordinates {{
            {' '.join(zipped_coords)}
            }};""")




            lines.append(f"""
            \\end{{axis}}
        \\end{{tikzpicture}}%
            """)
            with open(os.path.join(OUT_DIR, f"{os.path.basename(name)}_ap_{'_'.join(conf)}.tikz"), "w") as f:
                f.writelines(lines)

def get_oracle():
    full_data, full_data_test = None, None
    with ZipFile("data.zip") as archive:
        seed_everything(7)
        # full_data: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "train",
        #                                                                          to_device=device)
        full_data_test: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "2019")

    oracle = Oracle(full_data_test)
    if full_data is not None:
        oracle.add_data(full_data)
    return oracle

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

if __name__ == "__main__":
    CLI(main)


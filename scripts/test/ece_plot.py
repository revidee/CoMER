import math
from typing import Dict, List, Tuple
import ntpath
from zipfile import ZipFile

import numpy as np
import torch
from jsonargparse import CLI
import matplotlib.pyplot as plt
import matplotlib as mpl
from pytorch_lightning import seed_everything

from comer.datamodules import Oracle
from comer.datamodules.crohme import extract_data_entries, get_splitted_indices
from comer.utils import ECELoss
from comer.utils.conf_measures import score_ori, score_sum, score_bimin, score_bisum, score_rev_sum
from comer.utils.utils import Hypothesis

SEP: str = ','


def main():
    device = torch.device('cpu')
    with ZipFile("data.zip") as archive:
        seed_everything(7)

        calc_ece = True
        calc_name = "plots_test_bimin"

        if calc_ece:

            full_data: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "train",
                                                                                     to_device=device)
            full_data_test: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "2019",
                                                                                          to_device=device)
            oracle = Oracle(full_data)
            oracle.add_data(full_data_test)
            all_hyps: List[Dict[str, Hypothesis]] = []

            all_hyps.append(torch.load("../hyps_s_35_new_test_bimin.pt",
                                       map_location=torch.device('cpu')))
            all_hyps.append(torch.load("../hyps_s_35_new_original_ts_ce.pt",
                                       map_location=torch.device('cpu')))
            # all_hyps.append(torch.load("../hyps_s_35_new_original_ts_ece.pt",
            #                            map_location=torch.device('cpu')))
            # all_hyps.append(torch.load("../hyps_s_35_new_original_ts_both.pt",
            #                            map_location=torch.device('cpu')))

            ece = ECELoss()

            def hyp_to_triplet_ori(fname_and_hyp: Tuple[str, Hypothesis]):
                fname, hyp = fname_and_hyp
                hyp_len = len(hyp.seq)
                return (math.exp(score_ori(hyp)) if hyp_len > 0 else 0, hyp.seq, oracle.get_gt_indices(fname))

            def hyp_to_triplet_avg(fname_and_hyp: Tuple[str, Hypothesis]):
                fname, hyp = fname_and_hyp
                hyp_len = len(hyp.seq)
                return (math.exp(score_sum(hyp) / hyp_len) if hyp_len > 0 else 0, hyp.seq, oracle.get_gt_indices(fname))

            def hyp_to_triplet_rev_avg(fname_and_hyp: Tuple[str, Hypothesis]):
                fname, hyp = fname_and_hyp
                hyp_len = len(hyp.seq)
                return (
                math.exp(score_rev_sum(hyp) / hyp_len) if hyp_len > 0 else 0, hyp.seq, oracle.get_gt_indices(fname))

            def hyp_to_triplet_biavg(fname_and_hyp: Tuple[str, Hypothesis]):
                fname, hyp = fname_and_hyp
                hyp_len = len(hyp.seq)
                return (math.exp((score_sum(hyp) + score_rev_sum(hyp) / 2) / hyp_len) if hyp_len > 0 else 0, hyp.seq, oracle.get_gt_indices(fname))

            def hyp_to_triplet_bimin(fname_and_hyp: Tuple[str, Hypothesis]):
                fname, hyp = fname_and_hyp
                hyp_len = len(hyp.seq)
                return (math.exp(score_bimin(hyp)) if hyp_len > 0 else 0, hyp.seq, oracle.get_gt_indices(fname))

            def hyp_to_triplet_mult(fname_and_hyp: Tuple[str, Hypothesis]):
                fname, hyp = fname_and_hyp
                hyp_len = len(hyp.seq)
                return (math.exp(score_sum(hyp)) if hyp_len > 0 else 0, hyp.seq, oracle.get_gt_indices(fname))

            def hyp_to_triplet_bimult(fname_and_hyp: Tuple[str, Hypothesis]):
                fname, hyp = fname_and_hyp
                hyp_len = len(hyp.seq)
                return (math.exp(score_bisum(hyp)) if hyp_len > 0 else 0, hyp.seq, oracle.get_gt_indices(fname))

            scoring_fns = [
                hyp_to_triplet_ori,
                hyp_to_triplet_biavg,
                # hyp_to_triplet_rev_avg,
                hyp_to_triplet_bimin,
                hyp_to_triplet_mult,
                hyp_to_triplet_bimult
            ]

            plots = [
                [
                    ece.get_plot_bins(map(scoring_fn, hyps.items())) for scoring_fn in scoring_fns
                ] for hyps in all_hyps
            ]

            torch.save(plots, f"{calc_name}.pt")
        else:
            plots = torch.load(f"{calc_name}.pt", map_location=device)

        titles = [
            "ORI",
            "BIAVG",
            # "REV_AVG",
            "BIMIN",
            "MULT",
            "BIMULT"
        ]
        all_hyp_titles = [
            "Bimin Epoch 1",
            "TS CE",
            # "TS ECE",
            # "TS CE+ECE",
        ]

        fig, axes = plt.subplots(len(plots), len(plots[0]), sharey=True, sharex=True)
        fig.tight_layout(w_pad=-0.5, h_pad=0.)

        linewidth = 0.5
        edgecolor = "#444444"

        for row_idx, (data_per_scoring_fn) in enumerate(plots):
            for col_idx, (accs, confs, samples, (ece, tot_acc)) in enumerate(data_per_scoring_fn):
                ax = axes[row_idx][col_idx]
                width = (1 / 15)  # the width of the bars: can also be len(x) sequence
                for i in range(len(accs)):
                    acc_exp = (i / len(accs)) + (1 / (2 * len(accs)))
                    acc = accs[i]
                    if acc > acc_exp:
                        # Accuracy was higher than expected,
                        # Orange = Underconfidence
                        ax.bar(i * width, acc_exp,
                               color=mpl.colormaps["Blues"](((i * width) / 2) + 0.5), width=width,
                               align='edge', edgecolor=edgecolor, linewidth=linewidth)
                        ax.bar(i * width, acc - acc_exp, color=mpl.colormaps["Greens"](0.8 - ((i * width) / 3)),
                               width=width, bottom=acc_exp, align='edge', edgecolor=edgecolor, linewidth=linewidth)
                    else:
                        # Accuracy was lower than expected,
                        # Red = Overconfidence
                        ax.bar(i * width, acc, color=mpl.colormaps["Blues"](((i * width) / 2) + 0.5),
                               width=width, align='edge', edgecolor=edgecolor, linewidth=linewidth)
                        ax.bar(i * width, acc_exp - acc, color=mpl.colormaps["Reds"](((i * width) / 2) + 0.25),
                               width=width, bottom=acc, align='edge', edgecolor=edgecolor, linewidth=linewidth)
                    ax.text(
                        i * width + width / 2,
                        0.05,
                        f"{samples[i]}",
                        fontdict={'fontsize': 7},
                        ha="center",
                        bbox={
                            "boxstyle": "Round, pad=0.05, rounding_size=0.1",
                            "color": (1, 1, 1, 0.8),
                        }
                    )
                ax.set_xticks([(((i + 1) / len(accs))) for i in range(len(accs))], labels=samples)
                # labels = [f"{samples_in_bin}" for (i, samples_in_bin) in enumerate(samples)]
                # labels = [f"{(i+1)/len(accs):.2f})\n{samples_in_bin}" for (i, samples_in_bin) in enumerate(samples)]
                # labels[-1] = f"1.0]\n{samples[-1]}"
                ax.set_xticklabels([f"{(((i + 1) / len(accs))):.2f}" for i in range(len(accs))], rotation=0,
                                   fontdict={'fontsize': 8})

                ax.text(.035, 0.85, f"ECE={ece * 100:.1f}", backgroundcolor=(0.5, 0.5, 0.5, 0.5))
                ax.set_xlim((0, 1.0))
                ax.set_ylim((0, 1.0))
        for col_idx, title in enumerate(titles):
            axes[0][col_idx].set_title(title)
            axes[len(all_hyp_titles) - 1][col_idx].set_xlabel("Confidence")
        for i, title in enumerate(all_hyp_titles):
            axes[i][0].set_ylabel(f"{title}\nExpRate-0")
        axes[0][0].text(0.05, 1.1, "Overconfidence", backgroundcolor="#e83429dd", ha="left")
        axes[0][len(titles) - 1].text(0.95, 1.1, "Underconfidence", backgroundcolor="#157f3bdd", ha="right")

        plt.show()


if __name__ == '__main__':
    CLI(main)

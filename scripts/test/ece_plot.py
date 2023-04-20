import math
from typing import Dict, List, Tuple, Callable
import ntpath
from zipfile import ZipFile

import matplotlib
import numpy as np
import torch
from jsonargparse import CLI
import matplotlib.pyplot as plt
import matplotlib as mpl
from pytorch_lightning import seed_everything

from comer.datamodules import Oracle
from comer.datamodules.crohme import extract_data_entries, get_splitted_indices
from comer.utils import ECELoss
from comer.utils.conf_measures import score_ori, score_sum, score_bimin, score_bisum, score_rev_sum, score_bi_avg, \
    score_min
from comer.utils.utils import Hypothesis

SEP: str = ','


def main(save: bool = False):
    device = torch.device('cpu')
    with ZipFile("data.zip") as archive:
        seed_everything(7)

        calc_ece = True
        calc_name = "ece_plot_35_ts_ece_bimin_test"

        font_size = 48
        dpi = 96
        # fig_size_px = (1920, 540)
        fig_size_px = (1920, 1080)

        models_in_rows = True

        def hyp_to_triplet_with_scoring(sfn: Callable[[Hypothesis], float], oracle: Oracle):
            def hyp_to_triplet(tuple: [str, Hypothesis]):
                seq_len = len(tuple[1].seq)
                return np.exp(sfn(tuple[1])) if seq_len else 0.0, tuple[1].seq, oracle.get_gt_indices(tuple[0])
            return hyp_to_triplet

        scoring_fns = [
            # ('ORI', score_ori),
            # ('MIN', score_min),
            ('BIMIN', score_bimin),
            # ('MULT', score_sum),
            # ('BIMULT', score_bisum)
        ]

        all_hyp_cps = [
            # ("35%", "../hyps_s_35_new_original_1_test.pt",),
            # ("35% CE", "../hyps_s_35_new_original_ts_ce_test.pt",),
            # ("35% ECE", "../hyps_s_35_new_original_ts_ece_test.pt",),
            # ("35% CE+ECE", "../hyps_s_35_new_original_ts_both_test.pt",),
            # ("50%", "../hyps_s_50_new_original_1_test.pt",),
            # ("100% Supervised", "../hyps_s_100_new_original_test.pt",),
            ("35% CE", "../hyps_s_35_new_original_ts_ce_test.pt"),
            # ("TS ECE", "../hyps_s_35_new_original_ts_ece_test.pt"),
            # ("CE + TS", "../hyps_s_35_new_original_ts_both.pt"),
            # ("LN + TS", "../hyps_s_35_new_t0_1_opt.pt"),
            # ("TS CE+ECE", "../hyps_s_35_new_original_ts_both.pt"),
        ]



        if calc_ece:

            full_data: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "train",
                                                                                     to_device=device)
            full_data_test: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "2019",
                                                                                          to_device=device)
            oracle = Oracle(full_data)
            oracle.add_data(full_data_test)
            all_hyps: List[Dict[str, Hypothesis]] = [torch.load(cp, map_location=torch.device('cpu')) for (_, cp) in all_hyp_cps]

            ece = ECELoss()

            plots = [
                [
                    ece.get_plot_bins(map(hyp_to_triplet_with_scoring(scoring_fn[1], oracle), hyps.items())) for scoring_fn in scoring_fns
                ] for hyps in all_hyps
            ]

            torch.save(plots, f"{calc_name}.pt")
        else:
            plots = torch.load(f"{calc_name}.pt", map_location=device)


        matplotlib.rcParams.update({'font.size': font_size})

        for font_file in matplotlib.font_manager.findSystemFonts():
            matplotlib.font_manager.fontManager.addfont(font_file)


        rows = len(plots)
        cols = len(plots[0])
        if not models_in_rows:
            rows, cols = cols, rows
        fig, axes = plt.subplots(rows, cols, sharey=True, sharex=True, figsize=(fig_size_px[0] / dpi, fig_size_px[1] / dpi))
        fig.tight_layout(w_pad=-0.5, h_pad=0., pad=1)

        plt.gcf().subplots_adjust(left=0.12, top=0.9)

        linewidth = 0.5
        edgecolor = "#444444"


        def get_ax(row_idx, col_idx):

            col = col_idx
            row = row_idx
            if not models_in_rows:
                col = row_idx
                row = col_idx

            if len(plots) == 1:
                if len(data_per_scoring_fn) == 1:
                    ax = axes
                else:
                    ax = axes[col_idx]
            else:
                if len(data_per_scoring_fn) == 1:
                    ax = axes[row_idx]
                else:
                    ax = axes[row_idx][col_idx]
            return ax

        for row_idx, (data_per_scoring_fn) in enumerate(plots):
            for col_idx, (accs, confs, samples, (ece, tot_acc)) in enumerate(data_per_scoring_fn):
                ax = get_ax(row_idx, col_idx)
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
                        # fontdict={'fontfamily': 'Iosevka'},
                        ha="center",
                        # bbox={
                        #     "boxstyle": "Round, pad=0.05, rounding_size=0.1",
                        #     "color": (1, 1, 1, 0.2),
                        # }
                    )
                ax.set_xticks([(((i + 1) / len(accs))) for i in range(0, len(accs), 2)])
                # labels = [f"{samples_in_bin}" for (i, samples_in_bin) in enumerate(samples)]
                # labels = [f"{(i+1)/len(accs):.2f})\n{samples_in_bin}" for (i, samples_in_bin) in enumerate(samples)]
                # labels[-1] = f"1.0]\n{samples[-1]}"
                ax.set_xticklabels(
                    [f"{(((i + 1) / len(accs))):.2f}" for i in range(0, len(accs), 2)], rotation=0
                    # , fontdict={'fontfamily': 'Iosevka'}
                )

                ax.text(.035, 0.85, f"ECE={ece * 100:.1f}", backgroundcolor=(0.5, 0.5, 0.5, 0.5))
                ax.set_xlim((0, 1.0))
                ax.set_ylim((0, 1.0))

        for col_idx, (title, _) in enumerate(scoring_fns):
            if models_in_rows:
                get_ax(0, col_idx).set_title(title)
                get_ax(len(all_hyp_cps) - 1, col_idx).set_xlabel("Konfidenz")
            else:
                get_ax(0, col_idx).set_ylabel(f"ExpRate-0 ({title})")

        for i, (title, _) in enumerate(all_hyp_cps):
            if models_in_rows:
                get_ax(i, 0).set_ylabel(f"ExpRate-0 ({title})")
            else:
                get_ax(0, i).set_title(title)
                get_ax(len(all_hyp_cps) - 1, i).set_xlabel("Konfidenz")

        first_bb = get_ax(0, 0).get_position()
        last_ax = get_ax(0, cols - 1) if models_in_rows else get_ax(cols - 1, 0)
        last_bb = last_ax.get_position()

        Y_PADDING_CONF = 12
        X_PADDING_CONF = 6
        get_ax(0, 0).text(first_bb.xmin + (X_PADDING_CONF / fig_size_px[0]), first_bb.ymax + (Y_PADDING_CONF / fig_size_px[1]), "Underconfidence", backgroundcolor="#157f3bdd", ha="left", va='bottom', transform=plt.gcf().transFigure)
        last_ax.text(last_bb.xmax - (X_PADDING_CONF / fig_size_px[0]), last_bb.ymax + (Y_PADDING_CONF / fig_size_px[1]), "Overconfidence", backgroundcolor="#e83429dd", ha="right", va='bottom', transform=plt.gcf().transFigure)

        if save:
            plt.savefig(f'{calc_name}.pdf', format='pdf')
        else:
            plt.show()


if __name__ == '__main__':
    CLI(main)

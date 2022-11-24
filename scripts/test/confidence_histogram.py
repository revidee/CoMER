import math
from typing import Dict
import ntpath

import numpy as np
from jsonargparse import CLI
import matplotlib.pyplot as plt
import matplotlib as mpl

SEP: str = ','


def main(
        metrics: str = './2014_FileMetrics.csv',
        stats: str = './2014_stats.txt',
        combined: str = './2014_combined.csv',
):
    stat_dict: Dict = {}
    with open(stats, "r") as stats_file:
        for lineNo, line in enumerate(stats_file.readlines()):
            name_end = line.find(SEP)
            if name_end == -1:
                print(f"(warn) (stats) failed to find sep at line {lineNo}")
                continue
            name = line[:name_end]
            score_start = line.rfind(SEP)
            if score_start == -1:
                print(f"(warn) (stats) failed to find score seperator at line {lineNo}")
                continue
            score = float(line[score_start + 1:])

            len_start = line.rfind(SEP, 0, score_start)
            seq_len = int(line[len_start + 1:score_start])

            seq = line[name_end + 1:len_start]
            stat_dict[name] = {
                "score": score,
                "seq_len": seq_len,
                "seq": seq
            }
    with open(metrics, "r") as metrics_file:
        for lineNo, line in enumerate(metrics_file.readlines()):
            if line.startswith("sep=") or line.startswith("File,"):
                continue

            splits = line.split(SEP)
            filepath = splits[0]  # File field

            filename_with_ext = ntpath.basename(filepath)
            ext_dot_idx = filename_with_ext.rfind(".")
            if ext_dot_idx == -1:
                print(f"(warn) (metrics) expected a file extension, but found none at line {lineNo}")
                continue
            filename = filename_with_ext[:ext_dot_idx]

            errors = int(splits[2])  # D_B field

            if not (filename in stat_dict):
                print(f"(warn) (metrics) stats not found for {filename}, from line {lineNo}")
                continue

            stat_dict[filename]["errors"] = errors

    with open(combined, "w") as combined_file:
        combined_file.write(f"sep={SEP}\n")
        min_score = float('Inf')
        max_score = float('-Inf')
        for filename, filestats in stat_dict.items():
            score = filestats['score'] / 2
            combined_file.write(f"{filename},{filestats['errors']},{score},{filestats['seq_len']}\n")
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score

        max_score = math.exp(max_score)
        min_score = math.exp(min_score)
        tot_range = max_score - min_score
        bins = 20
        bin_width = tot_range / bins
        bin_ranges = [(max_score - (bin_width * i), max_score - (bin_width * (i + 1))) for i in range(bins)]
        MAX_ERROR_TOL = 5
        exp_rates = [[] for i in range(MAX_ERROR_TOL)]
        for error_tol in range(MAX_ERROR_TOL):
            bins_correct = [0 for _ in range(bins)]
            bins_total = [0 for _ in range(bins)]
            for filename, filestats in stat_dict.items():
                score = filestats['score'] / 2
                errors = filestats['errors']
                bin_idx = int((max_score - math.exp(score)) // bin_width)
                if bin_idx >= bins:
                    bin_idx = bins - 1
                bins_total[bin_idx] += 1
                if errors <= error_tol:
                    bins_correct[bin_idx] += 1
            exp_rates[error_tol] = [0 if tot == 0 else (corr / tot) for (corr, tot) in zip(bins_correct, bins_total)]
            print("err-tol", exp_rates[error_tol])
            print("Correct: ".ljust(10), bins_correct)
            print("Total: ".ljust(10), bins_total)
            print("ExpRates:")
            for f, exp in zip(bin_ranges, exp_rates[error_tol]):
                print('[{:.2f}, {:.2f}): {:.2f}'.format(f[0], f[1], exp * 100))
            print()
        fig, ax = plt.subplots(figsize=(8, 4))
        plt_bins = ['[{:.2f}, {:.2f})'.format(f[0], f[1]) for f in bin_ranges]
        X = np.arange(bins)
        width = 1 / (MAX_ERROR_TOL + 1)
        offset = (1 - width) / 2
        for i in range(MAX_ERROR_TOL):
            ax.bar(X + i * width - offset, exp_rates[i], color=mpl.colormaps["viridis"](i * width + width), width=width, align='edge')
        ax.set_xticks(X)
        ax.set_xticklabels(plt_bins, rotation=45)
        ax.legend([f"{i}-Error" for i in range(MAX_ERROR_TOL)])
        ax.set_ylabel("ExpRate")
        ax.set_xlabel("Confidence")
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    CLI(main)

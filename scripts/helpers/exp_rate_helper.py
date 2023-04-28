from os import path
from typing import List

import numpy as np
from jsonargparse import CLI


BASE_DIR = ""
# extract ExpRate-0 from multiple versions and years and prints them below each other to create a table
def main(base: str, years=None):
    if years is None:
        # years = ["2014", "2016", "2019"]
        years = ["2016"]

    version_rows = [
        # [191, 193, 88]
        [128, 126, "25_ts", 66, "21_ts", 86],
        [259, 341, 313, 337, 274, 326],
        [260, 340, 314, 336, 182, 332],
        [261, 339, 315, 335, 183, 331],
        [262, 338, 316, 333, 185, 330],
        # [191, 193, 88],
        # [301, 310, 304],
        # [306, 311, 307],
        # [308, 312, 309],
        # [88, 88, 88, 228, 228, 228],
        # [171, 290, 285, 241, 237, 231],
        # [300, 298, 280, 242, 238, 232],
        # [299, 297, 279, 273, 239, 233],
        # [289, 294, 277, 275, 240, 234]
    ]
    row_titles = [
        "Baseline",
        "$\\tau=0.95$",
        "$\\tau=0.5$",
        "$\\tau=0.125$",
        "$\\tau=0.05$",
        # "$\\tau=0.05$",
        # "$\\tau_\\text{H}=0.125$",
        # "$\\tau_\\text{M}=0.125$",
        # "Syn. Baseline",
        # "$\\tau=0.5$",
        # "$\\tau=0.25$",
        # "$\\tau=0.125$",
        # "$\\tau=0.075$",
    ]
    # Oracle
    # version_rows = [
    #     # [16, 16, 25, 25, 89, 89],
    #     [140, 218, 208, 214, 248, 243],
    #     [222, 219, 209, 215, 249, 247],
    #     [223, 220, 211, 216, 250, 245],
    #     [224, 221, 213, 217, 251, 246],
    # ]
    # row_titles = [
    #     # "Baseline",
    #     "$0$",
    #     "$1$",
    #     "$2$",
    #     "$3$",
    # ]
    np_rows = []
    curr_row = []
    for row_idx, (title, row) in enumerate(zip(row_titles, version_rows)):
        for j, version in enumerate(row):
            for i, year in enumerate(years):
                try:
                    with open(path.join(BASE_DIR, base, f"version_{version}", f"{year}.txt")) as f:
                        lines = f.readlines()
                        exp_zero = float(lines[1].split()[-1])
                        curr_row.append(exp_zero)
                except FileNotFoundError:
                    curr_row.append(0.0)
        np_rows.append(np.array(curr_row))
        curr_row = []
    all_exps = np.vstack(np_rows)
    max_row_idx_per_col = np.argmax(all_exps, axis=0)
    for row_idx, (title, row) in enumerate(zip(row_titles, version_rows)):
        print(f'{title} & ', flush=True, end='')
        for col_idx in range(len(all_exps[row_idx])):
            if max_row_idx_per_col[col_idx] == row_idx and len(row_titles) > 1:
                print(f"$\\mathbf{{{all_exps[row_idx][col_idx]:.1f}}}$", flush=True, end='')
            else:
                print(f"${all_exps[row_idx][col_idx]:.1f}$", flush=True, end='')
            if col_idx is not len(all_exps[row_idx]) - 1:
                print(" & ", flush=True, end='')
            elif row_idx is not len(version_rows) - 1:
                print(' \\\\ \\hline')
        print()



if __name__ == "__main__":
    CLI(main)
import os

import numpy as np
from jsonargparse import CLI
import scipy.stats

def main(base_folder: str, prefix: str):
    _, subdirs, _ = next(os.walk(base_folder))
    stats = {
        "2014.txt": {
            0: [],
            1: [],
            2: [],
            3: []
        },
        "2016.txt": {
            0: [],
            1: [],
            2: [],
            3: []
        },
        "2019.txt": {
            0: [],
            1: [],
            2: [],
            3: []
        },
    }
    for subdir in subdirs:
        if subdir.find(prefix) == 0 and (subdir[len(prefix):]).isdigit():
            for year_filename in stats.keys():
                fpath = os.path.join(os.path.join(base_folder, subdir), year_filename)
                with open(fpath, "r") as f:
                    lines = f.readlines()
                    for i in range(4):
                        line = lines[i + 1]
                        line.rindex(" ")
                        exp_rate = float(line[line.rindex(" ") + 1:])
                        stats[year_filename][i].append(exp_rate)

    header = f" 2014".center(15) + f"2016".center(15) + f"2019".center(15)
    lines = [
        "0",
        "1",
        "2",
        "3"
    ]

    for year_filename, all_rates in stats.items():
        for err_tol, exp_rates in all_rates.items():
            mean, conf = mean_confidence_interval(exp_rates)
            lines[err_tol] += f"{mean:.2f}±{conf:.2f}".center(15)
    print(header)
    for line in lines:
        print(line)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

if __name__ == "__main__":
    CLI(main)

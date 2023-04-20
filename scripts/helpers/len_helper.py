import re
from os import path
from typing import List
from zipfile import ZipFile

import numpy as np
import torch
from jsonargparse import CLI

from comer.datamodules import Oracle
from comer.datamodules.crohme import extract_data_entries

BASE_DIR = ""

# Load File Metrics and compute all correct sequences per-length, for all crohme test datasets
# outputs correct-per-len and total-per-len
def main(lenfile: str):
    device = torch.device("cpu")
    with ZipFile("../../data.zip") as archive:
        # full_data: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "train", to_device=device)
        full_data_test: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "2019", to_device=device)
        full_data_test1: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "2014", to_device=device)
        full_data_test2: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "2016", to_device=device)
    oracle = Oracle(full_data_test)
    oracle.add_data(full_data_test1)
    oracle.add_data(full_data_test2)

    correct_lens = np.zeros((200,), dtype=int)
    total_lens = np.zeros((200,), dtype=int)

    for year in [2014, 2016, 2019]:
        file = path.join(lenfile, f"{year}_FileMetrics.csv")
        with open(file) as f:
            for line in f.readlines():
                match = re.search("\/([a-zA-Z_0-9]*).lg,\s(In)?[cC]orrect", line)
                if match is None:
                    continue
                groups = match.groups()
                fname = groups[0]
                corr = match.groups()[1] is None

                gt = oracle.get_gt_indices(fname)
                if len(gt) > 199:
                    continue
                total_lens[len(gt)] += 1
                if corr:
                    correct_lens[len(gt)] += 1
    print(correct_lens.tolist())
    print(total_lens.tolist())

if __name__ == "__main__":
    CLI(main)
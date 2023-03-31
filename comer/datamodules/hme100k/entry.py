from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Dict
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Grayscale

from comer.datamodules.crohme import DataEntry
from comer.datamodules.hme100k.extract import get_hme_subsets
from comer.datamodules.hme100k.vocab import vocab

to_grayscale = Grayscale(num_output_channels=1)
def unzip_file(filename, label, path):
    img: Image.Image = to_grayscale(Image.open(path).copy())
    return DataEntry(filename, np.asarray(img), False, label, None)

def extract_data_entries(main_path: str,
                         prefix: str,
                         limit: Optional[int] = None,
                         subsets: Optional[Dict[str, set[str]]] = None,
                         ) -> 'np.ndarray[Any, np.dtype[DataEntry]]':
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        prefix (str): train or test split
        limit (int): optionally, limit the amount of returned items

    Returns:
        Data: list of tuple of image and formula
    """
    with open(f'{main_path}/{prefix}_labels.txt', encoding='utf-8') as f:
        captions = f.readlines()

        data: List[DataEntry] = []
        compatible_sets = dict()
        total_original = 0
        all_labels = dict()
        if subsets is not None:
            for (name, s) in subsets.items():
                total_original += len(s)
                compatible_sets[name] = []

        for idx, line in enumerate(captions):
            tmp: List[str] = line.strip().split()
            label: List[str] = tmp[1:]

            # filter incompatible tokens
            if not vocab.is_label_compatible(label):
                continue

            file_name: str = tmp[0]
            stipped_file_name: str = tmp[0][:-4]
            all_labels[file_name] = label
            # add file to available sets, to keep distribution the same
            if subsets is not None:
                for (name, s) in subsets.items():
                    if stipped_file_name in s:
                        compatible_sets[name].append(file_name)
                        break


        if limit is not None:
            # limit to X samples, while keeping the distribution of original test samples the same
            seed = torch.initial_seed()
            if seed > 2**32 - 1:
                seed = 7
            np.random.seed(seed)
            picked_labels = dict()

            for (name, s) in subsets.items():
                limit_per_subset = math.floor(len(s) * limit / total_original)
                compatible_sets[name] = np.array(compatible_sets[name])
                np.random.shuffle(compatible_sets[name])
                for fname in compatible_sets[name][:limit_per_subset]:
                    picked_labels[fname] = all_labels[fname]

            all_labels = picked_labels
        printed = 0
        done = 0
        with ProcessPoolExecutor(10) as exe:
            # unzip each file from the archive
            futures = [exe.submit(unzip_file, file_name, label, f"{main_path}/{prefix}_images/{file_name}") for (file_name, label) in all_labels.items()]
            # register the progress indicator callback
            for future in as_completed(futures):
                # retrieve the result
                data.append(future.result())
                done += 1
                if (done * 100) / len(all_labels) > printed:
                    print(f"Loading {prefix}: {printed}%", end='\r', flush=True)
                    printed += 1
            #
        # for i, (file_name, label) in enumerate(all_labels.items()):
        #     inner_file_path = f"{prefix}_images/{file_name}"
        #     with archive.open(inner_file_path, "r") as f:
        #         img: Image.Image = to_grayscale(Image.open(f).copy())
        #         data.append(DataEntry(file_name, img, False, label, None))
        #     if (i * 100) / len(all_labels) > printed:
        #         print(f"Loading {prefix}: {printed}%", end='\r', flush=True)
        #         printed += 1
        # print()
        #
        # logging.info(f"Extract data from: {prefix}, with data size: {len(data)}")

        return np.array(data)

if __name__ == '__main__':
    train, test, sets = get_hme_subsets(ZipFile('C:/Users/marca/Desktop/Master/HME100K.zip'))
    extract_data_entries(test, 'test', 10, sets)
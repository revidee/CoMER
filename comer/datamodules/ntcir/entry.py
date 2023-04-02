from __future__ import annotations

import logging
from typing import Optional, List
from zipfile import ZipFile

import numpy as np
from PIL import Image
from torchvision.transforms import Grayscale

from comer.datamodules.crohme import DataEntry, vocab


def extract_data_entries(archive: ZipFile,
                         prefix: str,
                         used_vocab=vocab,
                         ) -> 'np.ndarray[Any, np.dtype[DataEntry]]':
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        prefix (str): dir name in archive zip (eg: train, test)

    Returns:
        Data: list of tuple of image and formula
    """
    with archive.open(f"{prefix}/caption.txt", "r") as f:
        captions = f.readlines()
    data: List[DataEntry] = []
    to_grayscale = Grayscale(num_output_channels=1)
    for line in captions:
        tmp: List[str] = line.decode().strip().split()
        file_name: str = tmp[0]
        label: List[str] = tmp[1:]

        if not used_vocab.is_label_compatible(label):
            continue

        with archive.open(f"{prefix}/{file_name}.png", "r") as f:
            # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
            img = Image.open(f)
            img = np.asarray(to_grayscale(img))
            data.append(DataEntry(file_name, img, False, label, None))

    logging.info(f"Extract data from: {prefix}, with data size: {len(data)}")

    return np.array(data)
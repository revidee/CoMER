from dataclasses import dataclass
from typing import List, Any, Optional
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor


@dataclass
class DataEntry:
    file_name: str
    image: Image
    label: List[str]


def extract_data_entries(archive: ZipFile, dir_name: str,
                         to_device: Optional[torch.device] = None,
                         max_size: Optional[int] = None,
                         random_seed: Optional[int] = None) -> 'np.ndarray[Any, np.dtype[DataEntry]]':
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    with archive.open(f"data/{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    data: List[DataEntry] = []
    for line in captions:
        tmp: List[str] = line.decode().strip().split()
        file_name: str = tmp[0]
        label: List[str] = tmp[1:]
        with archive.open(f"data/{dir_name}/img/{file_name}.bmp", "r") as f:
            # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
            img: Image.Image = Image.open(f).copy()

            # Directly move the image to a target device.
            # This is needed to call this method from a CPU context (i.e. when doing CLI inference / benching)
            if to_device is not None:
                img = ToTensor()(img).to(device=to_device)
        data.append(DataEntry(file_name, img, label))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return np.array(data)

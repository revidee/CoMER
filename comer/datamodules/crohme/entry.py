import logging
from dataclasses import dataclass
from typing import List, Any, Optional, Union
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor


@dataclass
class DataEntry:
    file_name: str
    image: Image
    is_partial: bool
    label: Union[List[str], None]
    label_r2l: Union[List[str], None] = None


def extract_data_entries(archive: ZipFile, dir_name: str,
                         to_device: Optional[torch.device] = None,
                         max_size: Optional[int] = None,
                         random_seed: Optional[int] = None,
                         # file_ending: str = "png",
                         # folder_prefix: str = "",
                         # folder_infix: str = "",
                         folder_infix: str = "/img",
                         file_ending: str = "bmp",
                         folder_prefix: str = "data/"
                         ) -> 'np.ndarray[Any, np.dtype[DataEntry]]':
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    with archive.open(f"{folder_prefix}{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    if max_size is not None:
        captions = captions[0:max_size]
    data: List[DataEntry] = []
    for line in captions:
        tmp: List[str] = line.decode().strip().split()
        file_name: str = tmp[0]
        label: List[str] = tmp[1:]
        with archive.open(f"{folder_prefix}{dir_name}{folder_infix}/{file_name}.{file_ending}", "r") as f:
            # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
            img: Image.Image = Image.open(f).copy()

            # Directly move the image to a target device.
            # This is needed to call this method from a CPU context (i.e. when doing CLI inference / benching)
            if to_device is not None:
                img = ToTensor()(img).to(device=to_device)
        data.append(DataEntry(file_name, img, False, label, None))

    logging.info(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return np.array(data)

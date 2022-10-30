from dataclasses import dataclass
from typing import TypedDict, List
from zipfile import ZipFile

from PIL import Image


@dataclass
class DataEntry:
    file_name: str
    image: Image
    label: List[str]


def extract_data_entries(archive: ZipFile, dir_name: str) -> List[DataEntry]:
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
        data.append(DataEntry(file_name, img, label))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data
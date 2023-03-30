from typing import Optional, List
from zipfile import ZipFile

from comer.datamodules.utils.get_imagesize import get_image_metadata_from_bytesio, get_image_size_from_bytesio


def extract_data_entries(archive: ZipFile,
                         prefix: str,
                         ) -> 'np.ndarray[Any, np.dtype[DataEntry]]':
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        prefix (str): train or test split
        limit (int): optionally, limit the amount of returned items

    Returns:
        Data: list of tuple of image and formula
    """
    print("start")
    with archive.open(f"{prefix}_labels.txt", "r") as f:
        captions = f.readlines()

        data: List[DataEntry] = []
        printed = 0
        for idx, line in enumerate(captions):
            tmp: List[str] = line.decode().strip().split()
            file_name: str = tmp[0]
            label: List[str] = tmp[1:]

            inner_file_path = f"{prefix}_images/{file_name}"
            info = archive.getinfo(inner_file_path)

            with archive.open(inner_file_path, "r") as f:
                w, h = get_image_size_from_bytesio(f, info.file_size)
                if (idx / len(captions)) * 100 > printed:
                    print(f"{printed}%")
                    printed += 1
    print("done")

if __name__ == "__main__":
    extract_data_entries(ZipFile("A:\\Masterabeit\\hme100k\\train.zip"), "train")
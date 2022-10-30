from dataclasses import dataclass
from typing import List, Tuple, Callable
from zipfile import ZipFile

import numpy as np
from torch import FloatTensor, LongTensor

from comer.datamodule.crohme import DataEntry, extract_data_entries


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


# A BatchTuple represents a single batch which contains 3 lists of equal length (batch-len)
# [file_names, images, labels]
BatchTuple = Tuple[List[str], List[np.ndarray], List[List[str]]]

# change according to your GPU memory
MAX_SIZE = 32e4


def build_batches_from_entries(
        data: List[DataEntry],
        batch_size: int,
        batch_imagesize: int = MAX_SIZE,
        maxlen: int = 200,
        max_imagesize: int = MAX_SIZE,
) -> List[BatchTuple]:
    curr_fname_batch: List[str] = []
    curr_feature_batch: List[np.ndarray] = []
    curr_label_batch: List[List[str]] = []

    total_fname_batches: List[List[str]] = []
    total_feature_batches: List[List[np.ndarray]] = []
    total_label_batches: List[List[List[str]]] = []

    biggest_image_size = 0
    get_entry_image_pixels: Callable[[DataEntry], int] = lambda x: x.image.size[0] * x.image.size[1]
    data.sort(key=get_entry_image_pixels)

    i = 0
    for entry in data:
        size = get_entry_image_pixels(entry)
        image_arr = np.array(entry.image)
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(entry.label) > maxlen:
            print("label", i, "length bigger than", maxlen, "ignore")
        elif size > max_imagesize:
            print(
                f"image: {entry.file_name} size: {image_arr.shape[0]} x {image_arr.shape[1]} = {size} bigger than {max_imagesize}, ignore"
            )
        else:
            if batch_image_size > batch_imagesize or i == batch_size:
                # a batch is full, add it to the "batch"-list and reset the current batch with the new entry.
                total_fname_batches.append(curr_fname_batch)
                total_feature_batches.append(curr_feature_batch)
                total_label_batches.append(curr_label_batch)
                # reset current batch
                i = 0
                biggest_image_size = size
                curr_fname_batch = []
                curr_feature_batch = []
                curr_label_batch = []
            # add the entry to the current batch
            curr_fname_batch.append(entry.file_name)
            curr_feature_batch.append(image_arr)
            curr_label_batch.append(entry.label)
            i += 1

    # add last batch if it isn't empty
    if len(curr_fname_batch) > 0:
        total_fname_batches.append(curr_fname_batch)
        total_feature_batches.append(curr_feature_batch)
        total_label_batches.append(curr_label_batch)
    print("total ", len(total_feature_batches), "batch data loaded")
    return list(
        # Zips batches into a 3-Tuple Tuple[ List[str] , List[np.ndarray], List[List[str]] ]
        #                        Per batch:  file_names, images          , labels
        zip(
            total_fname_batches,
            total_feature_batches,
            total_label_batches
        )
    )


def build_dataset(
        archive: ZipFile,
        folder: str,
        batch_size: int
) -> List[BatchTuple]:
    return build_batches_from_entries(extract_data_entries(archive, folder), batch_size)

from dataclasses import dataclass
from typing import List, Tuple, Callable, Any
from zipfile import ZipFile

import numpy as np
import torch
from PIL.Image import Image
from torch import FloatTensor, LongTensor

from .entry import extract_data_entries, DataEntry
from .vocab import vocab


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    labels: List[List[int]]  # [b, l]
    is_labeled: bool
    src_idx: int

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            labels=self.labels,
            is_labeled=self.is_labeled,
            src_idx=self.src_idx
        )


# A BatchTuple represents a single batch which contains 3 lists of equal length (batch-len)
# [file_names, images, labels, is_labeled]
BatchTuple = Tuple[List[str], List['np.ndarray'], List[List[str]], bool, int]


# Creates a Batch of (potentially) annotated images which pads & masks the images, s.t. they fit into a single tensor.
def create_batch_from_lists(file_names: List[str], images: List['np.ndarray'], labels: List[List[str]], is_labled: bool, src_idx: int, remove_unlabeled: bool = False) -> Batch:
    assert (len(file_names) == len(images) == len(images))

    filtered_images: List['np.ndarray'] = []
    filtered_file_names: List[str] = []

    filtered_heights_x = []
    filtered_widths_x = []
    filtered_labels_as_indices = []

    for i, label in enumerate(labels):
        if not remove_unlabeled or len(label) > 0:
            filtered_im = images[i]
            filtered_images.append(filtered_im)
            filtered_labels_as_indices.append(vocab.words2indices(label))
            filtered_file_names.append((file_names[i]))
            filtered_heights_x.append(filtered_im.size(1))
            filtered_widths_x.append(filtered_im.size(2))


    n_samples = len(filtered_images)
    max_height_x = max(filtered_heights_x)
    max_width_x = max(filtered_widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, img in enumerate(filtered_images):
        x[idx, :, : filtered_heights_x[idx], : filtered_widths_x[idx]] = img
        x_mask[idx, : filtered_heights_x[idx], : filtered_widths_x[idx]] = 0

    return Batch(filtered_file_names, x, x_mask, filtered_labels_as_indices, is_labled, src_idx)


# change according to your GPU memory
MAX_SIZE = int(32e4)


def build_batch_split_from_entries(
        data: 'np.ndarray[Any, np.dtype[DataEntry]]',
        batch_size: int,
        batch_imagesize: int = MAX_SIZE,
        maxlen: int = 200,
        max_imagesize: int = MAX_SIZE,
        unlabeled_pct: float = 0,
        sorting_mode: int = 0  # 0 = nothing, 1 = random, 2 = sorted
) -> Tuple[List[BatchTuple], List[BatchTuple]]:
    total_len = len(data)

    idx_order = np.arange(total_len, dtype=int)

    if sorting_mode == 2:
        is_pil_image = isinstance(data[0].image, Image)
        if is_pil_image:
            get_entry_image_pixels: Callable[[DataEntry], int] = lambda x: x.image.size[0] * x.image.size[1]
        else:
            get_entry_image_pixels: Callable[[DataEntry], int] = lambda x: x.image.size(1) * x.image.size(2)
        idx_order = np.argsort(
            np.vectorize(get_entry_image_pixels)(data)
        )

    if mode == 1:
        np.random.seed(torch.initial_seed())
        np.random.shuffle(idx_order)

    if unlabeled_pct < 0:
        unlabeled_pct = 0
    if unlabeled_pct > 1:
        unlabeled_pct = 1

    labeled_end = int(total_len * (1 - unlabeled_pct))

    return (
        # labeled batches
        build_batches_from_samples(
            data[idx_order[:labeled_end]],
            batch_size,
            batch_imagesize,
            maxlen,
            max_imagesize,
            is_labled=True
        ),
        # unlabeled batches
        build_batches_from_samples(
            data[idx_order[labeled_end:]],
            batch_size,
            batch_imagesize,
            maxlen,
            max_imagesize,
            is_labled=False
        ),
    )


def build_batches_from_samples(
        data: 'np.ndarray[Any, np.dtype[DataEntry]]',
        batch_size: int,
        batch_imagesize: int = MAX_SIZE,
        maxlen: int = 200,
        max_imagesize: int = MAX_SIZE,
        is_labled: bool = True,
        include_last_only_full: bool = False
) -> List[BatchTuple]:
    if data.shape[0] == 0:
        return list()
    next_batch_file_names: List[str] = []
    next_batch_images: List['np.ndarray'] = []
    next_batch_labels: List[List[str]] = []

    total_fname_batches: List[List[str]] = []
    total_feature_batches: List[List['np.ndarray']] = []
    total_label_batches: 'List[List[List[str]]]' = []
    total_is_label_batches: List[bool] = []

    biggest_image_size = 0
    is_pil_image = isinstance(data[0].image, Image)
    if is_pil_image:
        get_entry_image_pixels: Callable[[DataEntry], int] = lambda x: x.image.size[0] * x.image.size[1]
    else:
        get_entry_image_pixels: Callable[[DataEntry], int] = lambda x: x.image.size(1) * x.image.size(2)

    # Sort the data entries via numpy by total pixel count and use the sorted indices to create a sorted array-view.
    data_sorted: 'np.ndarray[Any, np.dtype[DataEntry]]' = data[
        np.argsort(
            np.vectorize(get_entry_image_pixels)(data)
        )
    ]

    i = 0

    for entry in data_sorted:
        size = get_entry_image_pixels(entry)
        if is_pil_image:
            image_arr = np.array(entry.image)
        else:
            image_arr = entry.image
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(entry.label) > maxlen:
            print("label", i, "length bigger than", maxlen, "ignore")
        elif size > max_imagesize:
            if is_pil_image:
                print(
                    f"image: {entry.file_name} size: {image_arr.shape[0]} x {image_arr.shape[1]} = {size} bigger than {max_imagesize}, ignore"
                )
            else:
                print(
                    f"image: {entry.file_name} size: {image_arr.size(0)} x {image_arr.size(1)} = {size} bigger than {max_imagesize}, ignore"
                )
        else:
            if batch_image_size > batch_imagesize or i == batch_size:
                # a batch is full, add it to the "batch"-list and reset the current batch with the new entry.
                total_fname_batches.append(next_batch_file_names)
                total_feature_batches.append(next_batch_images)
                total_label_batches.append(next_batch_labels)
                total_is_label_batches.append(is_labled)
                # reset current batch
                i = 0
                biggest_image_size = size
                next_batch_file_names = []
                next_batch_images = []
                next_batch_labels = []
            # add the entry to the current batch
            next_batch_file_names.append(entry.file_name)
            next_batch_images.append(image_arr)
            if is_labled:
                next_batch_labels.append(entry.label)
            else:
                next_batch_labels.append([])
            i += 1

    # add last batch if it isn't empty
    if len(next_batch_file_names) > 0 and (not include_last_only_full or len(next_batch_file_names) == batch_size):
        total_fname_batches.append(next_batch_file_names)
        total_feature_batches.append(next_batch_images)
        total_label_batches.append(next_batch_labels)
        total_is_label_batches.append(is_labled)

    print(len(total_feature_batches), f"batches loaded (labled: {is_labled})")
    return list(
        # Zips batches into a 4-Tuple Tuple[ List[str] , List[np.ndarray], List[List[str]], bool ]
        #                        Per batch:  file_names, images          , labels           is_labeled
        zip(
            total_fname_batches,
            total_feature_batches,
            total_label_batches,
            total_is_label_batches,
            list(range(len(total_is_label_batches)))
        )
    )



def build_dataset(
        archive: ZipFile,
        folder: str,
        batch_size: int,
        unlabeled_pct: float = 0,
        sorting_mode: int = 0,  # 0 = nothing, 1 = random, 2 = sorted
) -> Tuple[List[BatchTuple], List[BatchTuple]]:
    return build_batch_split_from_entries(extract_data_entries(archive, folder), batch_size,
                                          unlabeled_pct=unlabeled_pct, sorting_mode=sorting_mode)

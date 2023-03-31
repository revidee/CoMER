import logging
import math
from dataclasses import dataclass
from typing import List, Tuple, Callable, Any, Union
from zipfile import ZipFile

import numpy as np
import torch
from PIL.Image import Image
from torch import FloatTensor, LongTensor

from .entry import extract_data_entries, DataEntry
from .vocab import vocab

# (is_partial, label_l2r, label_r2l),
MaybePartialLabel = Tuple[bool, Union[List[str], None], Union[List[str], None]]
MaybePartialLabelWithIndices = Tuple[bool, Union[List[int], None], Union[List[int], None]]
@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    labels: List[MaybePartialLabelWithIndices]  # [b, l]
    unlabeled_start: int
    src_idx: int
    unfiltered_size: int

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            labels=self.labels,
            unlabeled_start=self.unlabeled_start,
            src_idx=self.src_idx,
            unfiltered_size=self.unfiltered_size
        )

# A BatchTuple represents a single batch which contains 3 lists of equal length (batch-len)
# [file_names, images, labels unlabeled_start, src_idx]
BatchTuple = Tuple[List[str], List[Image], List[MaybePartialLabel], int, int]


# Creates a Batch of (potentially) annotated images which pads & masks the images, s.t. they fit into a single tensor.
def create_batch_from_lists(file_names: List[str], images: List['np.ndarray'], labels: List[MaybePartialLabel], is_labled: bool, src_idx: int, remove_unlabeled: bool = False) -> Batch:
    assert (len(file_names) == len(images) == len(images))

    filtered_images: List['np.ndarray'] = []
    filtered_file_names: List[str] = []

    filtered_heights_x = []
    filtered_widths_x = []
    filtered_labels_as_indices: List[MaybePartialLabelWithIndices] = []

    for i, label in enumerate(labels):
        if not remove_unlabeled \
                or (label[1] is not None and len(label[1]) > 0) \
                or (label[2] is not None and len(label[2]) > 0):
            filtered_im = images[i]
            filtered_images.append(filtered_im)
            filtered_labels_as_indices.append((
                label[0],
                vocab.words2indices(label[1]) if label[1] is not None else None,
                vocab.words2indices(label[2]) if label[2] is not None else None
            ))
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

    return Batch(filtered_file_names, x, x_mask, filtered_labels_as_indices, is_labled, src_idx, len(file_names))


# change according to your GPU memory
MAX_SIZE = int(32e4)


def get_splitted_indices(
        data: 'np.ndarray[Any, np.dtype[DataEntry]]',
        unlabeled_pct: float = 0,
        sorting_mode: int = 0  # 0 = nothing, 1 = random, 2 = sorted
):
    total_len = len(data)

    idx_order = np.arange(total_len, dtype=int)

    if sorting_mode == 2 or sorting_mode == 3:
        is_pil_image = isinstance(data[0].image, Image)
        if is_pil_image:
            get_entry_image_pixels: Callable[[DataEntry], int] = lambda x: x.image.size[0] * x.image.size[1]
        else:
            get_entry_image_pixels: Callable[[DataEntry], int] = lambda x: x.image.size(1) * x.image.size(2)
        idx_order = np.argsort(
            np.vectorize(get_entry_image_pixels)(data)
        )
        if sorting_mode == 3:
            idx_order = np.flip(idx_order)

    if sorting_mode == 1:
        np.random.seed(torch.initial_seed())
        np.random.shuffle(idx_order)

    if unlabeled_pct < 0:
        unlabeled_pct = 0
    if unlabeled_pct > 1:
        unlabeled_pct = 1

    labeled_end = int(total_len * (1 - unlabeled_pct))

    return idx_order[:labeled_end], idx_order[labeled_end:]


def build_batch_split_from_entries(
        data: 'np.ndarray[Any, np.dtype[DataEntry]]',
        batch_size: int,
        batch_imagesize: int = MAX_SIZE,
        maxlen: int = 200,
        max_imagesize: int = MAX_SIZE,
        unlabeled_pct: float = 0,
        sorting_mode: int = 0  # 0 = nothing, 1 = random, 2 = sorted (asc), 3 = sorted (dsc)
) -> Tuple[List[BatchTuple], List[BatchTuple]]:
    labeled_indices, unlabeled_indices = get_splitted_indices(data, unlabeled_pct=unlabeled_pct, sorting_mode=sorting_mode)

    return (
        # labeled batches
        build_batches_from_samples(
            data[labeled_indices],
            batch_size,
            batch_imagesize=batch_imagesize,
            max_imagesize=max_imagesize,
            maxlen=maxlen
        ),
        # unlabeled batches
        build_batches_from_samples(
            data[unlabeled_indices],
            batch_size,
            batch_imagesize=batch_imagesize,
            max_imagesize=max_imagesize,
            maxlen=maxlen
        ),
    )

def build_batches_from_samples(
        data: List[DataEntry],
        batch_size: int,
        batch_imagesize: int = MAX_SIZE,
        max_imagesize: int = MAX_SIZE,
        maxlen: int = 200,
        include_last_only_full: bool = False
) -> List[BatchTuple]:
    if data.shape[0] == 0:
        return list()
    next_batch_file_names: List[str] = []
    next_batch_images: List[Image] = []
    next_batch_labels: List[MaybePartialLabel] = []

    total_fname_batches: List[List[str]] = []
    total_feature_batches: List[List[Image]] = []
    total_label_batches: List[List[MaybePartialLabel]] = []
    total_unlabeled_start_batches: List[int] = []

    biggest_image_size = 0
    is_pil_image = isinstance(data[0].image, Image)
    if is_pil_image:
        get_entry_image_pixels: Callable[[DataEntry], int] = lambda x: x.image.size[0] * x.image.size[1]
    else:
        # Tensor on CPU
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
            logging.info(f"label {i} length bigger than {maxlen}, ignoring..")
        elif size > max_imagesize:
            if is_pil_image:
                logging.info(
                    f"image: {entry.file_name} size: {image_arr.shape[0]} x {image_arr.shape[1]} = {size} bigger than {max_imagesize}, ignore"
                            )
            else:
                logging.info(
                    f"image: {entry.file_name} size: {image_arr.size(0)} x {image_arr.size(1)} = {size} bigger than {max_imagesize}, ignore"
                )
        else:
            if batch_image_size > batch_imagesize or i == batch_size:
                # a batch is full, add it to the "batch"-list and reset the current batch with the new entry.
                total_fname_batches.append(next_batch_file_names)
                total_feature_batches.append(next_batch_images)
                total_label_batches.append(next_batch_labels)
                total_unlabeled_start_batches.append(len(next_batch_file_names))
                # reset current batch
                i = 0
                biggest_image_size = size
                next_batch_file_names = []
                next_batch_images = []
                next_batch_labels = []
            # add the entry to the current batch
            next_batch_file_names.append(entry.file_name)
            next_batch_images.append(image_arr)
            next_batch_labels.append((entry.is_partial, entry.label, entry.label_r2l))
            i += 1

    # add last batch if it isn't empty
    if len(next_batch_file_names) > 0 and (not include_last_only_full or len(next_batch_file_names) == batch_size):
        total_fname_batches.append(next_batch_file_names)
        total_feature_batches.append(next_batch_images)
        total_label_batches.append(next_batch_labels)
        total_unlabeled_start_batches.append(len(next_batch_file_names))

    logging.info(f"{len(total_feature_batches)} batches loaded")
    return list(
        # Zips batches into a 4-Tuple Tuple[ List[str] , List[np.ndarray], List[List[str]], bool ]
        #                        Per batch:  file_names, images          , labels           is_labeled
        zip(
            total_fname_batches,
            total_feature_batches,
            total_label_batches,
            total_unlabeled_start_batches,
            list(range(len(total_unlabeled_start_batches)))
        )
    )

def sort_data_entries_by_size(data: 'np.ndarray[Any, np.dtype[DataEntry]]'):
    if data.shape[0] == 0:
        return data
    is_pil_image = isinstance(data[0].image, Image)
    if is_pil_image:
        get_entry_image_pixels: Callable[[DataEntry], int] = lambda x: x.image.size[0] * x.image.size[1]
    else:
        # Tensor on CPU
        get_entry_image_pixels: Callable[[DataEntry], int] = lambda x: x.image.size(1) * x.image.size(2)

    # Sort the data entries via numpy by total pixel count and use the sorted indices to create a sorted array-view.
    return data[
        np.argsort(
            np.vectorize(get_entry_image_pixels)(data)
        )
    ]


def build_interleaved_batches_from_samples(
        labeled: 'np.ndarray[Any, np.dtype[DataEntry]]',
        unlabeled: 'np.ndarray[Any, np.dtype[DataEntry]]',
        batch_size: int,
) -> List[BatchTuple]:
    labeled_len = labeled.shape[0]
    unlabeled_len = unlabeled.shape[0]
    total = labeled_len + unlabeled_len
    if total == 0:
        return list()

    # Sort the data entries via numpy by total pixel count and use the sorted indices to create a sorted array-view.
    labeled_sorted = sort_data_entries_by_size(labeled)
    unlabeled_sorted = sort_data_entries_by_size(unlabeled)

    needed_batches = math.ceil(total / batch_size)

    unlabeled_per_batch = math.ceil(unlabeled_len / needed_batches)

    batches: List[BatchTuple] = []
    unlabeled_idx = 0
    labeled_idx = 0
    for i in range(needed_batches):
        unlabeled_end_idx = min(unlabeled_idx + unlabeled_per_batch, unlabeled_len)
        unlabeled_in_batch_len = unlabeled_end_idx - unlabeled_idx

        labeled_end_idx = min(labeled_idx + batch_size - unlabeled_in_batch_len, labeled_len)
        labeled_in_batch_len = labeled_end_idx - labeled_idx

        assert (unlabeled_in_batch_len + labeled_in_batch_len) > 0

        batch_entries: List[DataEntry] = labeled_sorted[labeled_idx:labeled_end_idx].tolist() + (
            [] if unlabeled_in_batch_len == 0 else unlabeled_sorted[unlabeled_idx:unlabeled_end_idx].tolist()
        )

        splitted_entries: List[Tuple[str, Image, MaybePartialLabel]] = [
                        (
                            entry.file_name,
                            entry.image,
                            (entry.is_partial, entry.label, entry.label_r2l)
                        ) for entry in batch_entries
                    ]
        partial_entries: Tuple[List[str], List[Image], List[MaybePartialLabel]] = list(zip(*splitted_entries))
        batches.append((
            partial_entries[0],
            partial_entries[1],
            partial_entries[2],
            labeled_in_batch_len,
            i
        ))

        unlabeled_idx = unlabeled_end_idx
        labeled_idx = labeled_end_idx
    return batches

def build_dataset(
        archive: ZipFile,
        folder: str,
        batch_size: int,
        unlabeled_pct: float = 0,
        sorting_mode: int = 0,  # 0 = nothing, 1 = random, 2 = sorted (asc), 3 = sorted (dsc)
) -> Tuple[List[BatchTuple], List[BatchTuple]]:
    return build_batch_split_from_entries(extract_data_entries(archive, folder), batch_size,
                                          unlabeled_pct=unlabeled_pct, sorting_mode=sorting_mode)

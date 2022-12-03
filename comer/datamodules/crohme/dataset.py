from typing import List

import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToPILImage

from comer.datamodules.crohme import BatchTuple
from comer.datamodules.utils.randaug import RandAugment
from comer.datamodules.utils.randaug_variants import fixmatch_modified
from comer.datamodules.utils.transforms import ScaleAugmentation, ScaleToLimitRange

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024


def build_aug_transform(aug_mode: str):
    trans_list = []
    if aug_mode == "weak":
        trans_list.append(ScaleAugmentation(K_MIN, K_MAX))
    elif aug_mode == "strong":
        trans_list.append(ToPILImage())
        trans_list.append(RandAugment(3))
    elif aug_mode == "strong_mod":
        trans_list.append(ToPILImage())
        trans_list.append(RandAugment(3, augments=fixmatch_modified()))

    trans_list += [
        ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
        tr.ToTensor(),
    ]
    return tr.Compose(trans_list)

class CROHMEDataset(Dataset):
    f"""
    Applies augmentations for all images inside a batch, just before it is used for training/validation/testing.
    The results of ``__getitem__`` are being fed into the collate function to create the final batch with
    the images padded to fit into a single tensor.
    """
    ds: List[BatchTuple]

    def __init__(self, ds: List[BatchTuple], aug_mode_labeled: str, aug_mode_unlabeled: str = "strong") -> None:
        super().__init__()
        self.ds = ds

        self.transform_labeled = build_aug_transform(aug_mode_labeled)
        self.transform_unlabeled = build_aug_transform(aug_mode_unlabeled)

    def __getitem__(self, idx):
        file_names, images, labels, unlabeled_start, src_idx = self.ds[idx]

        images = [
            self.transform_labeled(im) if i < unlabeled_start
            else self.transform_unlabeled(im) for (i, im) in enumerate(images)
        ]

        return file_names, images, labels, unlabeled_start, src_idx

    def __len__(self):
        return len(self.ds)
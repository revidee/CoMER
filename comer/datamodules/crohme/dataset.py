from typing import List

import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset

from comer.datamodules.crohme import BatchTuple
from comer.datamodules.utils.transforms import ScaleAugmentation, ScaleToLimitRange

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024


class CROHMEDataset(Dataset):
    f"""
    Applies augmentations for all images inside a batch, just before it is used for training/validation/testing.
    The results of ``__getitem__`` are being fed into the collate function to create the final batch with
    the images padded to fit into a single tensor.
    """
    ds: List[BatchTuple]

    def __init__(self, ds: List[BatchTuple], is_train: bool, scale_aug: bool) -> None:
        super().__init__()
        self.ds = ds

        trans_list = []
        if is_train and scale_aug:
            trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        trans_list += [
            ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
            tr.ToTensor(),
        ]
        self.transform = tr.Compose(trans_list)

    def __getitem__(self, idx):
        file_names, images, labels, is_labled = self.ds[idx]

        images = [self.transform(im) for im in images]

        return file_names, images, labels, is_labled

    def __len__(self):
        return len(self.ds)

from typing import List

import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset

from comer.datamodule.crohme import BatchTuple
from comer.datamodule.utils.transforms import ScaleAugmentation, ScaleToLimitRange

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024


class CROHMEDataset(Dataset):
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
        file_names, images, labels = self.ds[idx]

        images = [self.transform(im) for im in images]

        return file_names, images, labels

    def __len__(self):
        return len(self.ds)

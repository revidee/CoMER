# https://github.com/ildoonet/pytorch-randaugment/blob/48b8f509c4bbda93bbe733d98b3fd052b6e4c8ae/RandAugment/augmentations.py#L161
import random

from comer.datamodules.utils.randaug_variants import fixmatch_original


class RandAugment:
    def __init__(self, n, augments=None):
        if augments is None:
            augments = fixmatch_original()
        self.n = n
        self.augment_list = augments

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = minval + float(maxval - minval) * random.random()
            img = op(img, val)

        return img

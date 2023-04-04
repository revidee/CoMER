import logging
import os
import sys

import torch
from jsonargparse import CLI

from comer.datamodules import CROHMESupvervisedDatamodule

from pytorch_lightning import Trainer, seed_everything

from comer.datamodules.hme100k.variants.supervised import HMESupvervisedDatamodule
from model_lookups import AVAILABLE_MODELS


def main(
    cp: str,
    year: str = '2014',
    gpu: int = 0,
    aug: str = "",
    seed: int = 7,
    model: str = 'sup',
    data: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
    eval_batch_size: int = 4,
):
    assert model in AVAILABLE_MODELS
    usecrohme = not model.startswith("hme_")
    model = AVAILABLE_MODELS[model]
    seed_everything(seed)
    # generate output latex in result.zip
    trainer = Trainer(logger=False, accelerator='gpu', devices=[gpu])

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    dm = CROHMESupvervisedDatamodule(test_year=year, eval_batch_size=eval_batch_size, test_aug=aug, zipfile_path=data) if usecrohme else HMESupvervisedDatamodule(test_year=year, eval_batch_size=eval_batch_size, test_aug=aug, zipfile_path=data)

    device = torch.device(f'cuda:{gpu}')

    model = model.load_from_checkpoint(
        cp,
        test_suffix=f"{gpu}",
    ).to(device).eval()

    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    CLI(main)

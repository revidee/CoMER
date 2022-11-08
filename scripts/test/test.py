import os

import torch
from jsonargparse import CLI

from comer.datamodules import CROHMESupvervisedDatamodule
from comer.modules import CoMERSupervised
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)


def main(
    cp: str,
    year: str = '2014',
    gpu: int = 0,
):
    # generate output latex in result.zip
    trainer = Trainer(logger=False, accelerator='gpu', devices=[gpu])

    dm = CROHMESupvervisedDatamodule(test_year=year, eval_batch_size=4)

    device = torch.device(f'cuda:{gpu}')

    model = CoMERSupervised.load_from_checkpoint(cp).to(device).eval()

    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    CLI(main)

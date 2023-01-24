import os

import torch
from jsonargparse import CLI

from comer.datamodules import CROHMESupvervisedDatamodule
from comer.modules import CoMERSupervised, CoMERFixMatchInterleavedLogitNormTempScale, \
    CoMERFixMatchInterleavedTemperatureScaling
from pytorch_lightning import Trainer, seed_everything


class CoMERFixMatchInterleavedTemperatureScalingWithAdditionalHyperParams(CoMERFixMatchInterleavedTemperatureScaling):

    def __init__(self,
                 temperature: float,
                 patience: float,
                 monitor: str = "test",
                 **kwargs):
        super().__init__(**kwargs)

def main(
    cp: str,
    year: str = '2014',
    gpu: int = 0,
    aug: str = "weak",
    seed: int = 7
):
    seed_everything(seed)
    # generate output latex in result.zip
    trainer = Trainer(logger=False, accelerator='gpu', devices=[gpu])

    dm = CROHMESupvervisedDatamodule(test_year=year, eval_batch_size=4, test_aug=aug)

    device = torch.device(f'cuda:{gpu}')

    model = CoMERFixMatchInterleavedTemperatureScalingWithAdditionalHyperParams.load_from_checkpoint(
        cp,
        test_suffix=f"{gpu}",
        learning_rate_target=0.0001,
        steplr_steps=10
    ).to(device).eval()

    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    CLI(main)

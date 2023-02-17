
import torch
from jsonargparse import CLI

from comer.datamodules import CROHMESupvervisedDatamodule

from pytorch_lightning import Trainer, seed_everything
from model_lookups import AVAILABLE_MODELS


def main(
    cp: str,
    year: str = '2014',
    gpu: int = 0,
    aug: str = "",
    seed: int = 7,
    model: str = 'sup'
):
    assert model in AVAILABLE_MODELS
    model = AVAILABLE_MODELS[model]
    seed_everything(seed)
    # generate output latex in result.zip
    trainer = Trainer(logger=False, accelerator='gpu', devices=[gpu])

    dm = CROHMESupvervisedDatamodule(test_year=year, eval_batch_size=4, test_aug=aug)

    device = torch.device(f'cuda:{gpu}')

    model = model.load_from_checkpoint(
        cp,
        test_suffix=f"{gpu}",
    ).to(device).eval()

    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    CLI(main)

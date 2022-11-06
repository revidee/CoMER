import os

import typer
from comer.datamodules import CROHMESupvervisedDatamodule
from comer.modules import CoMERSupervised
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)


def main(version: str, test_year: str):
    # generate output latex in result.zip
    ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1
    ckp_path = os.path.join(ckp_folder, fnames[0])
    print(f"Test with fname: {fnames[0]}")

    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMESupvervisedDatamodule(test_year=test_year, eval_batch_size=4)

    model = CoMERSupervised.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    typer.run(main)

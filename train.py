from multiprocessing import freeze_support

from pytorch_lightning.cli import LightningCLI

from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER

if __name__ == '__main__':
    freeze_support()

    cli = LightningCLI(
        LitCoMER,
        CROHMEDatamodule,
        save_config_overwrite=True,
    )

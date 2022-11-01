from multiprocessing import freeze_support

from pytorch_lightning.cli import LightningCLI

if __name__ == '__main__':
    freeze_support()
    cli = LightningCLI(save_config_overwrite=True)

from abc import ABC, abstractmethod
from typing import Callable

from pytorch_lightning.trainer.progress import BatchProgress
from pytorch_lightning.utilities.fetching import AbstractDataFetcher


class UnlabeledLightningModule(ABC):

    @abstractmethod
    def unlabeled_full(self, data_fetcher: AbstractDataFetcher, start_batch: Callable, end_batch: Callable, dataloader_idx: int):
        return

    @abstractmethod
    def validation_unlabeled_step_end(self, output):
        return

    def validation_dataloader_end(self, output):
        pass

from collections import OrderedDict
from functools import partial
from typing import Any

from deprecate.utils import void
from pytorch_lightning.loops import EvaluationLoop

from .eval_epoch_loop import EvaluationWithUnlabeledEpochLoop


class EvaluationWithUnlabeledLoop(EvaluationLoop):
    """Loops over all dataloaders for evaluation."""

    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose=verbose)
        self.unlabeled_loop = EvaluationWithUnlabeledEpochLoop()

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Performs evaluation on one single dataloader."""
        void(*args, **kwargs)

        dataloader_idx = self.current_dataloader_idx
        dataloader = self.current_dataloader
        assert self._data_fetcher is not None
        self._data_fetcher.setup(
            dataloader,
            batch_to_device=partial(self.trainer._call_strategy_hook, "batch_to_device", dataloader_idx=dataloader_idx),
        )
        dl_max_batches = self._max_batches[dataloader_idx]

        kwargs = OrderedDict()
        if self.num_dataloaders > 1:
            kwargs["dataloader_idx"] = dataloader_idx
        if dataloader_idx == self.num_dataloaders - 1 and not self.trainer.sanity_checking:
            dl_outputs = self.unlabeled_loop.run(self._data_fetcher, dl_max_batches, kwargs)
        else:
            dl_outputs = self.epoch_loop.run(self._data_fetcher, dl_max_batches, kwargs)

        # store batch level output per dataloader
        self._outputs.append(dl_outputs)

        if not self.trainer.sanity_checking:
            # indicate the loop has run
            self._has_run = True

    def _reload_evaluation_dataloaders(self) -> None:
        """Reloads dataloaders if necessary."""
        dataloaders = None
        if self.trainer.testing:
            self.trainer.reset_test_dataloader()
            dataloaders = self.trainer.test_dataloaders
        elif self.trainer.val_dataloaders is None or self.trainer._data_connector._should_reload_val_dl:
            self.trainer.reset_val_dataloader()
            dataloaders = self.trainer.val_dataloaders
        if dataloaders is not None:
            self.epoch_loop._reset_dl_batch_idx(len(dataloaders))
            self.unlabeled_loop._reset_dl_batch_idx(len(dataloaders))

    def teardown(self) -> None:
        super().teardown()
        self.unlabeled_loop.teardown()

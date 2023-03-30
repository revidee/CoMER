from collections import OrderedDict
from typing import Any

from pytorch_lightning.loops import EvaluationLoop

from .eval_epoch_loop import EvaluationWithUnlabeledEpochLoop
from .eval_epoch_loop_end_hook import EvaluationWithRunEndHook


class EvaluationWithUnlabeledLoop(EvaluationLoop):
    """Loops over all dataloaders for evaluation."""

    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose=verbose)
        self.unlabeled_loop = EvaluationWithUnlabeledEpochLoop(self.get_self)
        self.epoch_loop = EvaluationWithRunEndHook()

    def get_self(self):
        return self

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Performs evaluation on one single dataloader."""
        dataloader_idx = self.current_dataloader_idx
        dataloader = self.current_dataloader

        def batch_to_device(batch: Any) -> Any:
            batch = self.trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
            batch = self.trainer._call_strategy_hook("batch_to_device", batch, dataloader_idx=dataloader_idx)
            return batch

        assert self._data_fetcher is not None
        self._data_fetcher.setup(dataloader, batch_to_device=batch_to_device)
        dl_max_batches = self._max_batches[dataloader_idx]

        kwargs = OrderedDict()
        if self.num_dataloaders > 1 and dataloader_idx == self.num_dataloaders - 1 and not self.trainer.sanity_checking:
            dl_outputs = self.unlabeled_loop.run(self._data_fetcher, dl_max_batches, kwargs)
        else:
            dl_outputs = self.epoch_loop.run(self._data_fetcher, dl_max_batches, kwargs)

        # store batch level output per dataloader
        self._outputs.append(dl_outputs)

    def sync_batches(self):
        self.epoch_loop._dl_batch_idx.clear()
        self.epoch_loop._dl_batch_idx.append(self.unlabeled_loop._dl_batch_idx)
        self.epoch_loop._dl_max_batches = self.unlabeled_loop._dl_max_batches
        self.epoch_loop.batch_progress.current.ready = self.unlabeled_loop.batch_progress.current.ready
        self.epoch_loop.batch_progress.current.started = self.unlabeled_loop.batch_progress.current.started
        self.epoch_loop.batch_progress.current.processed = self.unlabeled_loop.batch_progress.current.processed
        self.epoch_loop.batch_progress.current.completed = self.unlabeled_loop.batch_progress.current.completed
        self.epoch_loop.batch_progress.total.ready = self.unlabeled_loop.batch_progress.total.ready
        self.epoch_loop.batch_progress.total.started = self.unlabeled_loop.batch_progress.total.started
        self.epoch_loop.batch_progress.total.processed = self.unlabeled_loop.batch_progress.total.processed
        self.epoch_loop.batch_progress.total.completed = self.unlabeled_loop.batch_progress.total.completed

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

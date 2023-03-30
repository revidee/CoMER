from collections import OrderedDict
from functools import lru_cache
from typing import Any, Optional, Callable

from pytorch_lightning.loops import EvaluationEpochLoop
from pytorch_lightning.utilities.fetching import AbstractDataFetcher
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import STEP_OUTPUT


class EvaluationWithUnlabeledEpochLoop(EvaluationEpochLoop):
    """This is the loop performing the evaluation.
    """

    def __init__(self, get_parent: Callable[[], 'EvaluationWithUnlabeledLoop']) -> None:
        super().__init__()
        self.dataloader_idx = 0
        self.has_run = False
        self.data_fetcher = None
        self.get_parent = get_parent

    @property
    def done(self) -> bool:
        """Returns ``True`` if the current iteration count reaches the number of dataloader batches."""
        return self.has_run

    def on_run_start(  # type: ignore[override]
            self, data_fetcher: AbstractDataFetcher, dl_max_batches: int, kwargs: OrderedDict
    ) -> None:
        super().on_run_start(data_fetcher=data_fetcher, dl_max_batches=dl_max_batches, kwargs=kwargs)
        self.has_run = False

    def advance(  # type: ignore[override]
            self,
            data_fetcher: AbstractDataFetcher,
            dl_max_batches: int,
            kwargs: OrderedDict,
    ) -> None:
        """Calls the evaluation step with the corresponding hooks and updates the logger connector.

        Args:
            data_fetcher: iterator over the dataloader
            dl_max_batches: maximum number of batches the dataloader can produce
            kwargs: the kwargs passed down to the hooks.

        Raises:
            StopIteration: If the current batch is None
        """

        self.data_fetcher = data_fetcher

        # configure step_kwargs
        kwargs.setdefault("dataloader_idx", 0)
        kwargs = self._build_kwargs(kwargs, None, 0)

        self.dataloader_idx = kwargs.get("dataloader_idx", 0)

        # lightning module methods
        output = self._unlabeled_step(**kwargs)
        hook_name = "validation_unlabeled_step_end"
        # strategy_output = self.trainer._call_strategy_hook(hook_name, *args, **kwargs)
        self.trainer._call_lightning_module_hook(hook_name, output)

        # if not done by the unlabeled step, set the progress bar to done
        self.batch_progress.total.ready = dl_max_batches
        self.batch_progress.total.completed = dl_max_batches
        self.batch_progress.current.ready = dl_max_batches
        self.batch_progress.current.completed = dl_max_batches
        self.get_parent().sync_batches()

        self.has_run = True

        if not self.batch_progress.is_last_batch:
            # if fault tolerant is enabled and process has been notified, exit.
            self.trainer._exit_gracefully_on_signal()

    def _unlabeled_step(self, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """The evaluation step (validation_step or test_step depending on the trainer's state).

        Args:
            fetcher: The AbstractDataFetcher for the last dataloader
            batch_progress: To modify & increment the batch progress
            dataloader_idx: the index of the dataloader producing the current batch

        Returns:
            the outputs of the step
        """
        hook_name = "unlabeled_full"
        output = self.trainer._call_strategy_hook(hook_name, *kwargs.values())

        return output

    def start_batch(self, batch: Any, batch_idx: int):
        self.batch_progress.is_last_batch = self.data_fetcher.done
        self.batch_progress.increment_ready()
        self._on_evaluation_batch_start(batch=batch, batch_idx=batch_idx, dataloader_idx=self.dataloader_idx)
        self.batch_progress.increment_started()
        self.get_parent().sync_batches()

    def end_batch(self, batch: Any, batch_idx: int):
        self._evaluation_step_end(None)
        self.batch_progress.increment_processed()
        self._on_evaluation_batch_end(None, batch=batch, batch_idx=batch_idx, dataloader_idx=self.dataloader_idx)
        self.batch_progress.increment_completed()

        if not self.trainer.sanity_checking:
            self.trainer._logger_connector.update_eval_step_metrics(self._dl_batch_idx[self.dataloader_idx])
            self._dl_batch_idx[self.dataloader_idx] += 1
        self.get_parent().sync_batches()

    def _build_kwargs(self, kwargs: OrderedDict, batch: Any, batch_idx: int) -> OrderedDict:
        """Helper method to build the arguments for the current step.

        Args:
            kwargs: The kwargs passed down to the hooks.
            batch: The current batch to run through the step.

        Returns:
            The kwargs passed down to the hooks.
        """
        kwargs.update(fetcher=self.data_fetcher, start_batch=self.start_batch, end_batch=self.end_batch)
        # `dataloader_idx` should be last so we need to push these to the front
        kwargs.move_to_end("end_batch", last=False)
        kwargs.move_to_end("start_batch", last=False)
        kwargs.move_to_end("fetcher", last=False)
        return kwargs

    @lru_cache(1)
    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        """Whether the batch outputs should be stored for later usage."""
        model = self.trainer.lightning_module
        return is_overridden("validation_unlabeled_epoch_end", model)

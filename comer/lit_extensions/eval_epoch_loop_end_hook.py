from functools import lru_cache

from pytorch_lightning.loops import EvaluationEpochLoop
from pytorch_lightning.utilities.model_helpers import is_overridden

from comer.lit_extensions import UnlabeledLightningModule


class EvaluationWithRunEndHook(EvaluationEpochLoop):
    """This is the loop performing the evaluation.
    """

    def on_run_end(self):
        """Returns the outputs of the whole run."""
        outputs = self._outputs
        self.trainer._call_lightning_module_hook("validation_dataloader_end", outputs)

        return super().on_run_end()
    @lru_cache(1)
    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        """Whether the batch outputs should be stored for later usage."""
        model = self.trainer.lightning_module
        return is_overridden("validation_dataloader_end", model, UnlabeledLightningModule)

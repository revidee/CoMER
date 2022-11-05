from typing import Optional

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.types import STEP_OUTPUT


class DDPUnlabeledStrategy(DDPStrategy):
    """Strategy for multi-process single-device training on one or multiple nodes."""

    strategy_name = "ddp_unlabeled"

    def unlabeled_full(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            # print(self.lightning_module,
            #       [func for func in dir(self.lightning_module) if callable(getattr(self.lightning_module, func))])
            return self.lightning_module.unlabeled_full(*args, **kwargs)
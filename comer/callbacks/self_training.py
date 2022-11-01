from pytorch_lightning import Callback

from comer.datamodules.crohme import Batch


class SelfTrainingUpdater(Callback):
    def __init__(self, confidence_threshold: float = 0.5):
        super().__init__()
        print("callback inited!", confidence_threshold)
        self.state = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch: Batch, batch_idx, unused=0):
        self.state.append(outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        # access output using state
        all_outputs = self.state
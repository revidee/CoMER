# from typing import Optional, Tuple, List, Union
#
# import pytorch_lightning as pl
# from pytorch_lightning.trainer.connectors.data_connector import DataConnector, _DataLoaderSource
# from pytorch_lightning.trainer.states import RunningStage
# from pytorch_lightning.utilities import rank_zero_warn
# from pytorch_lightning.utilities.data import _replace_dunder_methods, _auto_add_worker_init_fn, has_len_all_ranks
# from pytorch_lightning.utilities.exceptions import MisconfigurationException
# from torch.utils.data import DataLoader
# from torchmetrics.utilities import apply_to_collection
#
# from .trainer import SelfTrainingTrainer
#


# Started to implement custom DataLoader for SelfTraining, but decided to opt for special-casing the last
# validator dataloader instead, s.t. the already-in-place DDP strategies for splitting the validation set can be used
# instead of making sure everything works fine with a new dataloader which needs to be split and loaded on N gpus.

# However, this might be useful at "some point" in the future, if the unlabeled processing gets it's own fitting step
# but this would entail even more modifications to pytorch lightning code...



#
# class SelfTrainingDataConnector(DataConnector):
#     def __init__(self, trainer: SelfTrainingTrainer, multiple_trainloader_mode: str = "max_size_cycle"):
#         super().__init__(trainer, multiple_trainloader_mode)
#         self._unlabeled_dataloader_source = _DataLoaderSource(None, "")
#
#     @property
#     def _should_reload_unlabeled_dl(self) -> bool:
#         """Check if validation dataloader should be reloaded."""
#         n_epochs = self.trainer.reload_dataloaders_every_n_epochs
#         return n_epochs and (
#                 self.trainer._last_unlabeled_dl_reload_epoch is None
#                 or self.trainer.current_epoch - self.trainer._last_unlabeled_dl_reload_epoch >= n_epochs
#         )
#
#     def attach_datamodule(
#             self, model: "pl.LightningModule", datamodule: Optional["pl.LightningDataModule"] = None
#     ) -> None:
#         self._unlabeled_dataloader_source = _DataLoaderSource(datamodule, "unlabeled_dataloader")
#         super().attach_datamodule(model, datamodule)
#
#     def _request_unlabeled_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
#         """Requests a dataloader from the given model by calling dataloader hooks corresponding to the given stage.
#
#         Returns:
#             The requested dataloader
#         """
#         source = getattr(self, "_unlabeled_dataloader_source")
#
#         with _replace_dunder_methods(DataLoader, "dataset"), _replace_dunder_methods(BatchSampler):
#             # under this context manager, the arguments passed to `DataLoader.__init__` will be captured and saved as
#             # attributes on the instance in case the dataloader needs to be re-instantiated later by Lightning.
#             # Also, it records all attribute setting and deletion using patched `__setattr__` and `__delattr__`
#             # methods so that the re-instantiated object is as close to the original as possible.
#             dataloader = source.dataloader()
#         if isinstance(dataloader, tuple):
#             dataloader = list(dataloader)
#         self.trainer.strategy.barrier("get_dataloaders")
#         return dataloader
#
#     def _reset_unlabeled_dataloader(
#             self, mode: RunningStage, model: Optional["pl.LightningModule"] = None
#     ) -> Tuple[List[Union[int, float]], List[DataLoader]]:
#         """Generic method to reset a dataloader for evaluation.
#
#         Args:
#             mode: The running stage of the ``Trainer``
#             model: The ``LightningModule`` if calling this outside of the trainer scope.
#
#         Returns:
#             Tuple (num_batches, dataloaders)
#         """
#
#         # always get the loaders first so we can count how many there are
#         dataloaders = self._request_unlabeled_dataloader()
#
#         if self.trainer.overfit_batches > 0:
#             dataloaders = self._resolve_overfit_batches(dataloaders, mode)
#
#         if not isinstance(dataloaders, list):
#             dataloaders = [dataloaders]
#
#         if any(dl is None for dl in dataloaders):
#             rank_zero_warn("One of given dataloaders is None and it will be skipped.")
#
#         # add samplers
#         dataloaders = [self._prepare_dataloader(dl, mode=mode) for dl in dataloaders if dl is not None]
#
#         # add worker_init_fn for correct seeding in worker processes
#         apply_to_collection(
#             dataloaders, dtype=DataLoader, function=_auto_add_worker_init_fn, rank=self.trainer.global_rank
#         )
#
#         loader_num_batches = []
#
#         # determine number of batches
#         module = model or self.trainer.lightning_module or self.datamodule
#         if len(dataloaders) != 0:
#             for i, dataloader in enumerate(dataloaders):
#                 orig_num_batches = num_batches = (
#                     len(dataloader) if has_len_all_ranks(dataloader, self.trainer.strategy, module) else float("inf")
#                 )
#
#                 if orig_num_batches == 0:
#                     loader_num_batches.append(orig_num_batches)
#                     continue
#
#                 self._worker_check(dataloader, f"unlabeled_dataloader {i}")
#
#                 # percent or num_steps
#                 limit_unlabeled_batches = getattr(self.trainer, f"limit_unlabeled_batches")
#
#                 # limit num batches either as a percent or num steps
#                 if isinstance(limit_unlabeled_batches, int):
#                     num_batches = min(orig_num_batches, limit_unlabeled_batches)
#                 elif isinstance(limit_unlabeled_batches, float) and orig_num_batches != float("inf"):
#                     num_batches = int(orig_num_batches * limit_unlabeled_batches)
#                 elif limit_unlabeled_batches != 1.0:
#                     raise MisconfigurationException(
#                         f"When using an `IterableDataset`, `Trainer(limit_unlabeled_batches)` must be"
#                         f" `1.0` or an int. An int specifies `num_unlabeled_batches` to use."
#                     )
#
#                 if (
#                         num_batches == 0
#                         and limit_unlabeled_batches > 0.0
#                         and isinstance(limit_unlabeled_batches, float)
#                         and orig_num_batches != float("inf")
#                 ):
#                     min_percentage = 1.0 / orig_num_batches
#                     raise MisconfigurationException(
#                         f"You requested to check {limit_unlabeled_batches} of the `unlabeled_dataloader` but"
#                         f" {limit_unlabeled_batches} * {orig_num_batches} < 1. Please increase the"
#                         f" `limit_unlabeled_batches` argument. Try at least"
#                         f" `limit_unlabeled_batches={min_percentage}`"
#                     )
#
#                 loader_num_batches.append(num_batches)
#
#         return loader_num_batches, dataloaders

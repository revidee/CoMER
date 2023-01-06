from pytorch_lightning import Trainer

from comer.lit_extensions.eval_loop import EvaluationWithUnlabeledLoop


class UnlabeledValidationExtraStepTrainer(Trainer):
    def __init__(self, unlabeled_val_loop: bool = False, **kwargs):
        super().__init__(**kwargs)
        if unlabeled_val_loop:
            self.fit_loop.epoch_loop.connect(self.fit_loop.epoch_loop.batch_loop, EvaluationWithUnlabeledLoop())
            self.fit_loop.trainer = self
            self.validate_loop = EvaluationWithUnlabeledLoop()
            self.validate_loop.trainer = self


    # Started to implement custom DataLoader for SelfTraining, but decided to opt for special-casing the last
    # validator dataloader instead, s.t. the already-in-place DDP strategies for splitting the validation set can be used
    # instead of making sure everything works fine with a new dataloader which needs to be split and loaded on N gpus.

    # However, this might be useful at "some point" in the future, if the unlabeled processing gets it's own fitting step
    # but this would entail even more modifications to pytorch lightning code...


    # def reset_unlabeled_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
    #     """Resets the unlabeled dataloader and determines the number of batches.
    #
    #     Args:
    #         model: The ``LightningModule`` if called outside of the trainer scope.
    #     """
    #     source = self._data_connector._unlabeled_dataloader_source
    #     pl_module = self.lightning_module or model
    #     has_step = is_overridden("unlabeled_step", pl_module)
    #     enable_unlabeled = self.limit_unlabeled_batches > 0
    #     if source.is_defined() and has_step and enable_unlabeled:
    #         # store epoch of dataloader reset for reload_dataloaders_every_n_epochs
    #         # it should not reload again if it has already reloaded during sanity_check
    #         if self.state.fn == TrainerFn.VALIDATING and (
    #                 (self.sanity_checking and self.fit_loop.epoch_loop._should_check_val_epoch())
    #                 or not self.sanity_checking
    #         ):
    #             self._last_unlabeled_dl_reload_epoch = self.current_epoch
    #
    #         self.num_unlabeled_batches, self.unlabeled_dataloaders = self._data_connector._reset_unlabeled_dataloader(
    #             RunningStage.VALIDATING, model=pl_module
    #         )
    #
    # @property
    # def unlabeled_loop(self) -> EvaluationUnlabeledLoop:
    #     return self._unlabeled_loop
    #
    # @unlabeled_loop.setter
    # def unlabeled_loop(self, loop: EvaluationUnlabeledLoop):
    #     """Attach a custom unlabeled loop to this Trainer.
    #
    #     It will run with
    #     :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate`. Note that this loop is different from the one
    #     running during training inside the :meth:`pytorch_lightning.trainer.trainer.Trainer.fit` call.
    #     """
    #     loop.trainer = loop
    #     self._unlabeled_loop = loop
    #
    # def _run_evaluate(self) -> _EVALUATE_OUTPUT:
    #     assert self.evaluating
    #
    #     # reload dataloaders
    #     self._evaluation_loop._reload_evaluation_dataloaders()
    #
    #     # reset trainer on this loop and all child loops in case user connected a custom loop
    #     self._evaluation_loop.trainer = self
    #
    #     with self.profiler.profile(f"run_{self.state.stage}_evaluation"), _evaluation_context(self.accelerator):
    #         eval_loop_results = self._evaluation_loop.run()
    #         self._unlabeled_loop.run()
    #
    #     # remove the tensors from the eval results
    #     for result in eval_loop_results:
    #         if isinstance(result, dict):
    #             for k, v in result.items():
    #                 if isinstance(v, Tensor):
    #                     result[k] = v.cpu().item()
    #     return eval_loop_results

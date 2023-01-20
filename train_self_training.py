from pytorch_lightning import seed_everything

from comer.lit_extensions import UnlabeledValidationExtraStepTrainer, DDPUnlabeledStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from comer.datamodules import CROHMESelfTrainingDatamodule
from comer.modules import CoMERSelfTraining
if __name__ == '__main__':
    seed_everything(7)

    trainer = UnlabeledValidationExtraStepTrainer(
        unlabeled_val_loop=True,
        accelerator='gpu',
        devices=[0, 1],
        strategy=DDPUnlabeledStrategy(find_unused_parameters=False),
        max_epochs=300,
        deterministic=True,
        reload_dataloaders_every_n_epochs=2,
        check_val_every_n_epoch=2,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1,
                            monitor='val_ExpRate/dataloader_idx_0',
                            mode='max',
                            filename='ep={epoch}-st={step}-valLoss={val_ExpRate/dataloader_idx_0:.4f}',
                            auto_insert_metric_name=False
                            ),
        ],
        precision=32
    )
    dm = CROHMESelfTrainingDatamodule(
        test_year='2019',
        val_year='2014',
        eval_batch_size=4,
        zipfile_path='data.zip',
        train_batch_size=8,
        num_workers=5,
        scale_aug=True
    )

    model: CoMERSelfTraining = CoMERSelfTraining.load_from_checkpoint(
        './bench/baseline_t112.ckpt',
        # Training (Supervised Tuning)
        learning_rate=0.00125,
        learning_rate_target=8e-5,
        steplr_steps=5,
        # Self-Training Params
        pseudo_labeling_threshold=0.985
    )

    trainer.fit(model, dm)

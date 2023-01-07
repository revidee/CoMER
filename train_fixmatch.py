from pytorch_lightning import seed_everything

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from comer.datamodules.crohme.variants.fixmatch_interleaved import CROHMEFixMatchInterleavedDatamodule
from comer.lit_extensions import UnlabeledValidationExtraStepTrainer, DDPUnlabeledStrategy
from comer.modules import CoMERFixMatchInterleavedLogitNormTempScale

if __name__ == '__main__':
    seed_everything(7)

    trainer = UnlabeledValidationExtraStepTrainer(
        unlabeled_val_loop=True,
        accelerator='gpu',
        devices=[6, 7],
        strategy=DDPUnlabeledStrategy(find_unused_parameters=False),
        max_epochs=300,
        deterministic=True,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1,
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
    dm = CROHMEFixMatchInterleavedDatamodule(
        test_year='2019',
        eval_batch_size=4,
        zipfile_path='data.zip',
        train_batch_size=8,
        num_workers=5,
        unlabeled_pct=0.65,
        train_sorting=1,
        unlabeled_strong_aug="weak",
        unlabeled_weak_aug=""
    )

    model: CoMERFixMatchInterleavedLogitNormTempScale = CoMERFixMatchInterleavedLogitNormTempScale.load_from_checkpoint(
        './lightning_logs/version_66/checkpoints/optimized_ts_0.5421.ckpt',
        strict=False,
        learning_rate=0.002,
        patience=20,
        pseudo_labeling_threshold=0.975,
        lambda_u=1.0,
        temperature=3.0,
        keep_old_preds=True,
        monitor="val_ExpRate/dataloader_idx_0",
        logit_norm_temp=0.1,
        th_optim_correct_weight=9,
        th_optim_sharpening=50
    )

    trainer.fit(model, dm)

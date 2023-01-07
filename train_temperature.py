from zipfile import ZipFile

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from comer.datamodules.crohme import CROHMEDataset, build_dataset
from comer.datamodules.crohme.variants.collate import collate_fn
from comer.datamodules.crohme.variants.fixmatch_interleaved import CROHMEFixMatchInterleavedDatamodule
from comer.lit_extensions import UnlabeledValidationExtraStepTrainer, DDPUnlabeledStrategy
from comer.modules.fixmatch_inter_logitnorm_ts import CoMERFixMatchInterleavedLogitNormTempScale

if __name__ == '__main__':
    seed_everything(7)

    trainer = UnlabeledValidationExtraStepTrainer(
        unlabeled_val_loop=True,
        accelerator='gpu',
        devices=[2, 3],
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
        precision=32,
        inference_mode=False
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
        './lightning_logs/version_69/checkpoints/epoch=207-step=54704-val_ExpRate=0.5513.ckpt',
        strict=False,
        learning_rate=0.0008,
        patience=20,
        pseudo_labeling_threshold=0.2,
        lambda_u=1.0,
        temperature=3.0,
        keep_old_preds=True,
        monitor="val_ExpRate/dataloader_idx_0",
        logit_norm_temp=0.05,
        th_optim_correct_weight=9,
        th_optim_sharpening=50
    )
    model.set_verbose_temp_scale_optim(True)

    with ZipFile("data.zip") as f:
        trainer.validate(model, DataLoader(
            CROHMEDataset(
                build_dataset(f, "2019", 4)[0],
                "",
                "",
            ),
            shuffle=False,
            num_workers=5,
            collate_fn=collate_fn,
        ))
        print("Re-evaluating with NLL/ECE optimized temperature...")
        trainer.validate(model, DataLoader(
            CROHMEDataset(
                build_dataset(f, "2019", 4)[0],
                "",
                "",
            ),
            shuffle=False,
            num_workers=5,
            collate_fn=collate_fn,
        ))
        print(trainer.logged_metrics)
        trainer.save_checkpoint(
            f'./lightning_logs/version_69/checkpoints/optimized_ts_{trainer.logged_metrics["val_ExpRate"]:.4f}.ckpt',
            False
        )



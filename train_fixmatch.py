from pytorch_lightning import seed_everything

from comer.datamodules.crohme.variants.fixmatch import CROHMEFixMatchDatamodule
from comer.datamodules.crohme.variants.fixmatch_inter_oracle import CROHMEFixMatchOracleDatamodule
from comer.datamodules.crohme.variants.fixmatch_interleaved import CROHMEFixMatchInterleavedDatamodule
from comer.lit_extensions import UnlabeledValidationExtraStepTrainer, DDPUnlabeledStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from comer.datamodules import CROHMESelfTrainingDatamodule
from comer.modules import CoMERSelfTraining
from comer.modules.fixmatch import CoMERFixMatch
from comer.modules.fixmatch_inter_oracle import CoMERFixMatchOracleInterleaved
from comer.modules.fixmatch_interleaved import CoMERFixMatchInterleaved

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
    dm = CROHMEFixMatchDatamodule(
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

    model: CoMERFixMatch = CoMERFixMatchInterleaved.load_from_checkpoint(
        './lightning_logs/version_25/checkpoints/epoch=293-step=154644-val_ExpRate=0.5488.ckpt',
        learning_rate=0.0008,
        patience=20,
        pseudo_labeling_threshold=0.9959,
        lambda_u=1.0
    )

    trainer.fit(model, dm)

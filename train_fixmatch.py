from pytorch_lightning import seed_everything

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from comer.datamodules import CROHMESupvervisedDatamodule
from comer.datamodules.crohme.variants.fixmatch_inter_oracle import CROHMEFixMatchOracleDatamodule
from comer.datamodules.crohme.variants.fixmatch_interleaved import CROHMEFixMatchInterleavedDatamodule
from comer.lit_extensions import UnlabeledValidationExtraStepTrainer, DDPUnlabeledStrategy
from comer.modules import CoMERFixMatchInterleavedLogitNormTempScale, CoMERFixMatchInterleavedTemperatureScaling
from comer.modules.fixmatch_inter_logitnorm_ts_oracle import CoMERFixMatchOracleInterleavedLogitNormTempScale


class CoMERFixMatchInterleavedTemperatureScalingWithAdditionalHyperParams(CoMERFixMatchInterleavedTemperatureScaling):

    def __init__(self,
                 temperature: float,
                 patience: float,
                 **kwargs):
        super().__init__(**kwargs)


if __name__ == '__main__':
    seed_everything(7)

    # monitor_suffix = '/dataloader_idx_0'
    monitor_suffix = ''

    trainer = UnlabeledValidationExtraStepTrainer(
        unlabeled_val_loop=True,
        accelerator='gpu',
        devices=[0, 1, 2, 3, 4, 5, 6, 7],
        strategy=DDPUnlabeledStrategy(find_unused_parameters=False),
        max_epochs=300,
        deterministic=True,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1,
                            monitor=f'val_ExpRate{monitor_suffix}',
                            mode='max',
                            filename=f'ep={{epoch}}-st={{step}}-valExpRate={{val_ExpRate{monitor_suffix}:.4f}}',
                            auto_insert_metric_name=False
                            ),
            ModelCheckpoint(save_top_k=1,
                            monitor=f'val_loss{monitor_suffix}',
                            mode='min',
                            filename=f'ep={{epoch}}-st={{step}}-valLoss={{val_loss{monitor_suffix}:.4f}}',
                            auto_insert_metric_name=False
                            ),
        ],
        precision=32
    )
    # dm = CROHMEFixMatchOracleDatamodule(
    dm = CROHMESupvervisedDatamodule(
        test_year='2019',
        val_year='2014',
        eval_batch_size=4,
        zipfile_path='data.zip',
        train_batch_size=8,
        num_workers=5,
        unlabeled_pct=0.85,
        train_sorting=1,
        # unlabeled_strong_aug="weak",
        # unlabeled_weak_aug=""
    )

    model: CoMERFixMatchInterleavedTemperatureScalingWithAdditionalHyperParams = CoMERFixMatchInterleavedTemperatureScalingWithAdditionalHyperParams.load_from_checkpoint(
        './lightning_logs/version_21/checkpoints/epoch=289-step=64960-val_ExpRate=0.3628.ckpt',
        strict=False,
        # Training (Supervised Tuning)
        learning_rate=0.00125,
        learning_rate_target=8e-5,
        steplr_steps=60,
        # Self-Training Params
        pseudo_labeling_threshold=0.3,
        lambda_u=1.0,
        # logit_norm_temp=0.1
    )

    trainer.fit(model, dm)

# Step-LR Values

# Supverised initial
# learning_rate=0.08,
# learning_rate_target=8e-5,
# steplr_steps=50


# Supverised Tuning
# learning_rate=0.00125,
# learning_rate_target=8e-5,
# steplr_steps=40,

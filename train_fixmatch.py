from pytorch_lightning import seed_everything

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from comer.datamodules import CROHMESupvervisedDatamodule
from comer.datamodules.crohme.variants.fixmatch_inter_oracle import CROHMEFixMatchOracleDatamodule
from comer.datamodules.crohme.variants.fixmatch_interleaved import CROHMEFixMatchInterleavedDatamodule
from comer.lit_extensions import UnlabeledValidationExtraStepTrainer, DDPUnlabeledStrategy
from comer.modules import CoMERFixMatchInterleavedLogitNormTempScale, CoMERFixMatchInterleavedTemperatureScaling, \
    CoMERFixMatchInterleavedFixedPctTemperatureScaling, CoMERFixMatchOracleInterleavedTempScale, \
    CoMERFixMatchInterleavedFixedPctLogitNormTempScale
from comer.modules.fixmatch_inter_logitnorm_ts_oracle import CoMERFixMatchOracleInterleavedLogitNormTempScale


# class CoMERFixMatchInterleavedTemperatureScalingWithAdditionalHyperParams(CoMERFixMatchInterleavedLogitNormTempScale):
class CoMERFixMatchInterleavedTemperatureScalingWithAdditionalHyperParams(CoMERFixMatchInterleavedLogitNormTempScale):

    def __init__(self,
                 temperature: float = 1.0,
                 patience: float = 20,
                 monitor: str = 'val_ExpRate',
                 **kwargs):
        super().__init__(**kwargs)


if __name__ == '__main__':
    seed_everything(7)

    monitor_suffix = '/dataloader_idx_0'
    # monitor_suffix = ''

    trainer = UnlabeledValidationExtraStepTrainer(
        unlabeled_val_loop=True,
        accelerator='gpu',
        devices=[4],
        strategy=DDPUnlabeledStrategy(find_unused_parameters=False),
        max_epochs=400,
        deterministic=True,
        reload_dataloaders_every_n_epochs=2,
        check_val_every_n_epoch=2,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1,
                            monitor=f'val_ExpRate{monitor_suffix}',
                            mode='max',
                            filename=f'ep={{epoch}}-st={{step}}-loss={{val_loss{monitor_suffix}:.4f}}-exp={{val_ExpRate{monitor_suffix}:.4f}}',
                            auto_insert_metric_name=False
                            ),
            ModelCheckpoint(save_top_k=1,
                            monitor=f'val_loss{monitor_suffix}',
                            mode='min',
                            filename=f'ep={{epoch}}-st={{step}}-loss={{val_loss{monitor_suffix}:.4f}}-exp={{val_ExpRate{monitor_suffix}:.4f}}',
                            auto_insert_metric_name=False
                            ),
        ],
        precision=32,
        sync_batchnorm=True
    )
    # dm = CROHMEFixMatchOracleDatamodule(
    dm = CROHMEFixMatchInterleavedDatamodule(
    # dm = CROHMESupvervisedDatamodule(
        test_year='2019',
        val_year='2019',
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
        # './lightning_logs/version_88/checkpoints/ep=145-st=32704-valExpRate=0.4987.ckpt',
        './lightning_logs/version_109/checkpoints/ep=245-st=55104-loss=0.4297-exp=0.5154.ckpt',
        # './lightning_logs/version_84/checkpoints/ep=143-st=16128-valExpRate=0.5146.ckpt',
        # './lightning_logs/version_77/checkpoints/epoch=41-step=236712-val_ExpRate=0.9099.ckpt',
        strict=False,
        # Training (Supervised Tuning)
        learning_rate=0.01,
        learning_rate_target=8e-4,
        steplr_steps=6,
        # Self-Training Params
        pseudo_labeling_threshold=0.45,
        keep_old_preds=False,  # Fixed-Percent Self-Training
        lambda_u=0.333,
        logit_norm_temp=0.1,
        partial_labeling_enabled=True,
        partial_labeling_only_below_normal_threshold=True,
        partial_labeling_min_conf=0.05,
        partial_labeling_std_fac=3.5,
        partial_labeling_std_fac_fade_conf_exp=2.0,

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

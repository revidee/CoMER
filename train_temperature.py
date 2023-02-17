import logging
import sys
from pathlib import Path
from zipfile import ZipFile

import torch.cuda
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from comer.datamodules.crohme import CROHMEDataset, build_dataset
from comer.datamodules.crohme.variants.collate import collate_fn
from comer.datamodules.crohme.variants.fixmatch_interleaved import CROHMEFixMatchInterleavedDatamodule
from comer.lit_extensions import UnlabeledValidationExtraStepTrainer, DDPUnlabeledStrategy
from comer.modules import CoMERFixMatchInterleavedFixedPctLogitNormTempScale, CoMERFixMatchInterleavedTemperatureScaling
from comer.modules.fixmatch_inter_logitnorm_ts import CoMERFixMatchInterleavedLogitNormTempScale

class CoMERFixMatchInterleavedTemperatureScalingWithAdditionHyperParams(CoMERFixMatchInterleavedTemperatureScaling):
    def __init__(self,
                 temperature: float,
                 patience: float,
                 # monitor: str,
                 **kwargs):
        super().__init__(**kwargs)

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    cps = [
        # ("./lightning_logs/version_64/checkpoints/epoch=177-step=46814-val_ExpRate=0.5079.ckpt"),
        # ("./lightning_logs/version_70/checkpoints/epoch=209-step=55230-val_ExpRate=0.5254.ckpt"),
        # ("./lightning_logs/version_65/checkpoints/epoch=239-step=63120-val_ExpRate=0.5463.ckpt"),
        # ("./lightning_logs/version_66/checkpoints/optimized_ts_0.5421.ckpt"),
        # ("./lightning_logs/version_66/checkpoints/epoch=291-step=76796-val_ExpRate=0.5338.ckpt"),
        # ("./lightning_logs/version_67/checkpoints/epoch=211-step=55756-val_ExpRate=0.5146.ckpt"),
        # ("./lightning_logs/version_68/checkpoints/epoch=257-step=67854-val_ExpRate=0.5013.ckpt"),
        # ("./lightning_logs/version_71/checkpoints/epoch=197-step=52074-val_ExpRate=0.5321.ckpt"),
        # ("./lightning_logs/version_25/checkpoints/epoch=293-step=154644-val_ExpRate=0.5488.ckpt"),
        # ("./lightning_logs/version_21/checkpoints/epoch=289-step=64960-val_ExpRate=0.3628.ckpt"),
        # ("./lightning_logs/version_128/checkpoints/epoch=234-step=178365-val_loss=0.3255.ckpt"),
        ("./lightning_logs/version_16/checkpoints/epoch=275-step=209484-val_ExpRate=0.5947.ckpt"),
        ("./lightning_logs/version_17/checkpoints/epoch=209-step=78120-val_ExpRate=0.5063.ckpt"),
    ]

    for cp_path in cps:
        seed_everything(7)
        trainer = UnlabeledValidationExtraStepTrainer(
            unlabeled_val_loop=True,
            accelerator='gpu',
            devices=[0],
            strategy=DDPUnlabeledStrategy(find_unused_parameters=False),
            deterministic=True,
            precision=32,
            inference_mode=False,
            enable_checkpointing=False,
            logger=False,
        )
        dm = CROHMEFixMatchInterleavedDatamodule(
            test_year='2019',
            val_year='2014',
            eval_batch_size=4,
            zipfile_path='data.zip',
            train_batch_size=8,
            num_workers=5,
            unlabeled_pct=0.85,
            train_sorting=1,
            unlabeled_strong_aug="weak",
            unlabeled_weak_aug=""
        )
        torch.cuda.empty_cache()
        p = Path(cp_path)
        if not p.exists():
            print(f"Checkpoint '{cp_path}' not found, skipping.")
            continue

        model: CoMERFixMatchInterleavedTemperatureScaling = CoMERFixMatchInterleavedTemperatureScaling.load_from_checkpoint(
            cp_path,
            strict=False,
            # Training
            learning_rate_target=8e-4,
            steplr_steps=10,
            # Self-Training
            pseudo_labeling_threshold=0.2,
            lambda_u=1.0,
            # logit_norm_temp=0.05,
        )
        model.set_verbose_temp_scale_optim(True)
        model.validation_global_pruning_overwrite = 'none'

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
            trainer.save_checkpoint(
                p.parent.joinpath(f'optimized_ts_{trainer.logged_metrics["val_ExpRate"]:.4f}.ckpt'),
                False
            )



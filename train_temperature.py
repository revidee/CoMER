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

if __name__ == '__main__':



    cps = [
        # ("./lightning_logs/version_64/checkpoints/epoch=177-step=46814-val_ExpRate=0.5079.ckpt"),
        # ("./lightning_logs/version_70/checkpoints/epoch=209-step=55230-val_ExpRate=0.5254.ckpt"),
        # ("./lightning_logs/version_65/checkpoints/epoch=239-step=63120-val_ExpRate=0.5463.ckpt"),
        ("./lightning_logs/version_66/checkpoints/optimized_ts_0.5421.ckpt"),
        ("./lightning_logs/version_66/checkpoints/epoch=291-step=76796-val_ExpRate=0.5338.ckpt"),
        # ("./lightning_logs/version_67/checkpoints/epoch=211-step=55756-val_ExpRate=0.5146.ckpt"),
        # ("./lightning_logs/version_68/checkpoints/epoch=257-step=67854-val_ExpRate=0.5013.ckpt"),
        # ("./lightning_logs/version_71/checkpoints/epoch=197-step=52074-val_ExpRate=0.5321.ckpt"),
        # ("./lightning_logs/version_25/checkpoints/epoch=293-step=154644-val_ExpRate=0.5488.ckpt"),
    ]

    for cp_path in cps:
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
            precision=32,
            inference_mode=False
        )
        dm = CROHMEFixMatchInterleavedDatamodule(
            test_year='2019',
            val_year='2014',
            eval_batch_size=4,
            zipfile_path='data.zip',
            train_batch_size=8,
            num_workers=5,
            unlabeled_pct=0.65,
            train_sorting=1,
            unlabeled_strong_aug="weak",
            unlabeled_weak_aug=""
        )
        torch.cuda.empty_cache()
        p = Path(cp_path)
        if not p.exists():
            print(f"Checkpoint '{cp_path}' not found, skipping.")
            continue

        model: CoMERFixMatchInterleavedLogitNormTempScale = CoMERFixMatchInterleavedLogitNormTempScale.load_from_checkpoint(
            cp_path,
            strict=False,
            # Self-Training
            pseudo_labeling_threshold=0.2,
            lambda_u=1.0,
            # logit_norm_temp=0.05,
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
            trainer.save_checkpoint(
                p.parent.joinpath(f'optimized_ts_{trainer.logged_metrics["val_ExpRate"]:.4f}.ckpt'),
                False
            )



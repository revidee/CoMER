import logging
import re
import sys
from datetime import datetime

from jsonargparse import CLI
import pprint

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from comer.datamodules import CROHMESupvervisedDatamodule
from comer.datamodules.crohme.variants.fixmatch_inter_oracle import CROHMEFixMatchOracleDatamodule
from comer.datamodules.crohme.variants.fixmatch_interleaved import CROHMEFixMatchInterleavedDatamodule
from comer.lit_extensions import UnlabeledValidationExtraStepTrainer, DDPUnlabeledStrategy

from comer.utils.conf_measures import CONF_MEASURES
from model_lookups import AVAILABLE_MODELS, POSSIBLE_CP_SHORTCUTS

LEARNING_PROFILES = {
    'initial': {
        'epochs': 400,
        'learning_rate': 0.08,
        'learning_rate_target': 8e-5,
        'steplr_steps': 8,
        'check_val_every_n_epoch': 1
    },
    'initial_val2': {
        'epochs': 400,
        'learning_rate': 0.08,
        'learning_rate_target': 8e-5,
        'steplr_steps': 8,
        'check_val_every_n_epoch': 2
    },
    'initial_bigger_steps': {
        'epochs': 400,
        'learning_rate': 0.15,
        'learning_rate_target': 8e-4,
        'steplr_steps': 5,
        'check_val_every_n_epoch': 2
    },
    'st': {
        'epochs': 400,
        'learning_rate': 0.02,
        'learning_rate_target': 8e-5,
        'steplr_steps': 8,
        'check_val_every_n_epoch': 2
    },
    'st2': {
        'epochs': 250,
        'learning_rate': 0.01,
        'learning_rate_target': 4e-4,
        'steplr_steps': 5,
        'check_val_every_n_epoch': 2
    }
}
PARTIAL_LABEL_PROFILES = {
    'high': {
        'partial_labeling_enabled': True,
        'partial_labeling_only_below_normal_threshold': True,
        'partial_labeling_min_conf': 0.00,
        'partial_labeling_std_fac': 0.0,
        'partial_labeling_std_fac_fade_conf_exp': 0.0,
    },
    'med': {
        'partial_labeling_enabled': True,
        'partial_labeling_only_below_normal_threshold': True,
        'partial_labeling_min_conf': 0.1,
        'partial_labeling_std_fac': 3.5,
        'partial_labeling_std_fac_fade_conf_exp': 2.0,
    },
    'low': {
        'partial_labeling_enabled': True,
        'partial_labeling_only_below_normal_threshold': True,
        'partial_labeling_min_conf': 0.3,
        'partial_labeling_std_fac': 5.5,
        'partial_labeling_std_fac_fade_conf_exp': 1.0,
    }
}

AVAILABLE_DATAMODULES = {
    'sup': CROHMESupvervisedDatamodule,
    'ora': CROHMEFixMatchOracleDatamodule,
    'fx': CROHMEFixMatchInterleavedDatamodule
}
# GLOBAL_PRUNING_THRESHOLDS_FOR_EPOCHS_PRESETS = {
#     'none': [],
#     'sup': [(15, 0.8), (30, 0.4), (60, 0.1), (400, 0.05)],
#     'st': [(30, 0.1), (400, 0.05)],
#     'st2': [(30, 0.3), (400, 0.15)],
#     'partial': [(400, 0.05)],
# }
VALID_GLOBAL_PRUNING_MODES = [
    'none',
    'sup',
    'st',
    'st2',
    'partial',
]

def main(
        gpu: int,
        cp: str = '',
        pct: float = None,
        lu: float = 1.0,
        keeppreds: bool = False,
        thresh: float = 1.0,
        pmthresh: float = None,
        pprof: str = '',
        model: str = 'sup',
        dm: str = 'sup',
        learn: str = 'initial',
        lntemp: float = 0.1,
        conf: str = 'ori',
        gprune: str = 'ori',
):

    assert model in AVAILABLE_MODELS
    assert dm in AVAILABLE_DATAMODULES
    assert learn in LEARNING_PROFILES
    assert conf in CONF_MEASURES
    assert gprune in VALID_GLOBAL_PRUNING_MODES;

    cp_addition = ''

    if cp in POSSIBLE_CP_SHORTCUTS:
        cp_addition = f'_from_{cp}'
        len_pct = re.findall('_(\d+)', cp)
        if len(len_pct) == 1:
            pct = float(len_pct[0]) / 100.0
        cp = POSSIBLE_CP_SHORTCUTS[cp]

    if pct is None:
        pct = 1.0

    assert pct > 0.0 and pct <= 1.0

    includes_st = dm != 'sup'
    is_ln = model.startswith('ln')
    is_partial = len(pprof) > 0

    if is_partial:
        assert pprof in PARTIAL_LABEL_PROFILES

    lu_addition = f'_lu{lu}' if lu != 1.0 else ''
    partial_addition = f'' if len(pprof) == 0 else f'_{pprof}{f"_{pmthresh}" if pmthresh is not None else ""}'
    st_addition = f'_{pct*100:.0f}_{thresh}' \
                  f'_{"keep" if keeppreds else "nokeep"}' \
                  f'{lu_addition}' \
                  f'{partial_addition}' if includes_st else ''

    date_time = datetime.now()
    timestamp = date_time.strftime("%m-%d-%H-%M-%S")

    logname = f"{model}_{dm}{st_addition}{cp_addition}__{timestamp}.log"


    model_class = AVAILABLE_MODELS[model]
    dm_class = AVAILABLE_DATAMODULES[dm]

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(
        logging.FileHandler(logname)
    )
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    seed_everything(7)

    # set params

    learning = dict(LEARNING_PROFILES[learn])

    monitor_suffix = '/dataloader_idx_0' if includes_st else ''

    trainer = UnlabeledValidationExtraStepTrainer(
        unlabeled_val_loop=True,
        accelerator='gpu',
        devices=[gpu],
        strategy=DDPUnlabeledStrategy(find_unused_parameters=False),
        max_epochs=learning['epochs'],
        deterministic=True,
        reload_dataloaders_every_n_epochs=learning['check_val_every_n_epoch'],
        check_val_every_n_epoch=learning['check_val_every_n_epoch'],
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1,
                            monitor=f'val_ExpRate{monitor_suffix}',
                            mode='max',
                            filename=f'ep={{epoch}}-st={{step}}-loss={{val_loss{monitor_suffix}:.4f}}-exp={{val_ExpRate{monitor_suffix}:.4f}}',
                            auto_insert_metric_name=False
                            ),
            # ModelCheckpoint(save_top_k=1,
            #                 monitor=f'val_loss{monitor_suffix}',
            #                 mode='min',
            #                 filename=f'ep={{epoch}}-st={{step}}-loss={{val_loss{monitor_suffix}:.4f}}-exp={{val_ExpRate{monitor_suffix}:.4f}}',
            #                 auto_insert_metric_name=False
            #                 ),
        ],
        precision=32,
        sync_batchnorm=True
    )
    # dm = CROHMEFixMatchOracleDatamodule(
    # dm = CROHMEFixMatchInterleavedDatamodule(
    dm = dm_class(
        test_year='2019',
        val_year='2019',
        eval_batch_size=4,
        zipfile_path='data.zip',
        train_batch_size=8,
        num_workers=5,
        unlabeled_pct=1.0-pct,
        train_sorting=1,
        # unlabeled_strong_aug="weak",
        # unlabeled_weak_aug=""
    )

    logging.info(f"Initiated Dataloader ({dm}) with {pct*100:.1f}% Labeled")

    new_model_kwargs = {
        'd_model': 256,
        # encoder,
        'growth_rate': 24,
        'num_layers': 16,
        # decoder,
        'nhead': 8,
        'num_decoder_layers': 3,
        'dim_feedforward': 1024,
        'dropout': 0.3,
        'dc': 32,
        # Fusion Coverage = both True
        'cross_coverage': True,
        'self_coverage': True,
        # beam search,
        'beam_size': 10,
        'max_len': 200,
        'alpha': 1.0,
        'early_stopping': False
    }

    kwargs = {
        'pseudo_labeling_threshold': thresh,
        'keep_old_preds': keeppreds,  # Fixed-Percent Self-Training
        'lambda_u': lu,
        'global_pruning_mode': gprune
    }

    if model != 'sup':
        kwargs["conf_fn"] = conf

    del learning["epochs"]
    del learning["check_val_every_n_epoch"]
    kwargs.update(**learning)
    if is_ln:
        kwargs['logit_norm_temp'] = lntemp
    if is_partial:
        kwargs.update(**PARTIAL_LABEL_PROFILES[pprof])
        if pmthresh is not None:
            kwargs["partial_labeling_min_conf"] = pmthresh

    if len(cp):
        kwargs['strict'] = False
        logging.info(f"Loading model (\"{model}\") from cp {cp if len(cp_addition) == 0 else cp_addition[6:]} kwargs: {pprint.pformat(kwargs)}")
        model: model_class = model_class.load_from_checkpoint(

            cp,
            **kwargs
        )
    else:
        kwargs.update(**new_model_kwargs)
        logging.info(f"Creating model (\"{model}\") kwargs: {pprint.pformat(kwargs)}")
        model = model_class(
            **kwargs
        )

    trainer.fit(model, dm)

if __name__ == '__main__':
    CLI(main)

# Step-LR Values

# Supverised initial
# learning_rate=0.08,
# learning_rate_target=8e-5,
# steplr_steps=50


# Supverised Tuning
# learning_rate=0.00125,
# learning_rate_target=8e-5,
# steplr_steps=40,

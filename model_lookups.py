from comer.modules import CoMERFixMatchInterleavedLogitNormTempScale, CoMERFixMatchInterleavedTemperatureScaling, \
    CoMERFixMatchInterleavedFixedPctTemperatureScaling, CoMERFixMatchOracleInterleavedTempScale, \
    CoMERFixMatchInterleavedFixedPctLogitNormTempScale, CoMERSupervised, CoMERFixMatchInterleaved, \
    CoMERFixMatchOracleInterleaved, CoMERFixMatchInterleavedFixedPct
from comer.modules.fixmatch_inter_logitnorm_ts_oracle import CoMERFixMatchOracleInterleavedLogitNormTempScale

AVAILABLE_MODELS = {
    'sup': CoMERSupervised,
    'fx': CoMERFixMatchInterleaved,
    'fx_fixed': CoMERFixMatchInterleavedFixedPct,

    'fx_ts': CoMERFixMatchInterleavedTemperatureScaling,
    'fx_ts_fixed': CoMERFixMatchInterleavedFixedPctTemperatureScaling,

    'fx_ora': CoMERFixMatchOracleInterleaved,
    'fx_ora_ts': CoMERFixMatchOracleInterleavedTempScale,

    'ln_ts': CoMERFixMatchInterleavedLogitNormTempScale,
    'ln_ts_fixed': CoMERFixMatchInterleavedFixedPctLogitNormTempScale,
    'ln_ora_ts': CoMERFixMatchOracleInterleavedLogitNormTempScale
}

POSSIBLE_CP_SHORTCUTS = {
    'syn': './lightning_logs/version_77/checkpoints/epoch=41-step=236712-val_ExpRate=0.9099.ckpt',
    's_15': './lightning_logs/version_89/checkpoints/epoch=363-step=81536-val_ExpRate=0.4078.ckpt',
    's_15_ln': './lightning_logs/version_86/checkpoints/epoch=165-step=37184-val_ExpRate=0.3344.ckpt',
    's_25': './lightning_logs/version_17/checkpoints/epoch=209-step=78120-val_ExpRate=0.5063.ckpt',
    's_25_ln': './lightning_logs/version_131/checkpoints/ep=268-st=100068-loss=0.5298-exp=0.4796.ckpt',

    's_35': './lightning_logs/version_25/checkpoints/epoch=293-step=154644-val_ExpRate=0.5488.ckpt',
    's_35_ln': './lightning_logs/version_66/checkpoints/epoch=291-step=76796-val_ExpRate=0.5338.ckpt',
    's_50': './lightning_logs/version_16/checkpoints/epoch=275-step=209484-val_ExpRate=0.5947.ckpt',
    's_50_ln': './lightning_logs/version_126/checkpoints/epoch=251-step=191268-val_ExpRate=0.6047.ckpt',
    's_75': './lightning_logs/version_14/checkpoints/epoch=209-step=238770-val_ExpRate=0.6230.ckpt',
    's_100': './lightning_logs/version_0/checkpoints/epoch=151-step=57151-val_ExpRate=0.6365.ckpt',

    'syn_15': './lightning_logs/version_88/checkpoints/ep=145-st=32704-valExpRate=0.4987.ckpt',
    'syn_15_ln': './lightning_logs/version_109/checkpoints/ep=245-st=55104-loss=0.4297-exp=0.5154.ckpt',
}
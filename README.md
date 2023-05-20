<div align="center">    
 
# CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition 
## With Semi-Supervised Learning Methods and HME100K Dataset support
 
 Original Paper
 
[![arXiv](https://img.shields.io/badge/arXiv-2207.04410-b31b1b.svg)](https://arxiv.org/abs/2207.04410)

 Master Thesis adding different Methods
 
 __to be published__
</div>

This repo expands the official implementation of [CoMER](https://github.com/Green-Wood/CoMER) and adds

- Support for PyTorch 1.13 and PyTorch Lightning 1.19
- Improved Beam-Search with Pruning Methods from [Freitag et al (Beam Search Strategies for Neural Machine Translation)](https://arxiv.org/abs/1702.01806)
  - With the addition of Constant Pruning
  - Improves inference speeds on CROHME19 Trainingdata roughly 7-fold
- Self-Training Methods like [FixMatch](https://arxiv.org/abs/2001.07685)
- Calibration Methods via learnable [Temperature Scaling](https://arxiv.org/abs/1706.04599) and [LogitNorm](https://arxiv.org/abs/2205.09310)
- Multiple new Confidence Measures for further improving Calibration
- [RandAug](https://arxiv.org/abs/1909.13719) with two augmentation lists, the modified version being better suited for long formulae
- Support for different vocabularies
- Support for synthetic Pre-Training with a generated NTCIR12 MathIR Dataset
- Support for [HME100K Dataset](https://arxiv.org/abs/2203.01601)
- A partial-labeling heuristic to replace a hard threshold while filtering generated pseudo-labels
- Multi-GPU Evaluation support
- Evaluation with Augmentations
- Tools \& Scripts to Visualize the data, Test the implementation and benchmark the modified beam-search

The Features are included in branches:
- [feature/ssl](https://github.com/revidee/CoMER/tree/feature/ssl), no helpers (visualization), no HME100K support
- [feature/ssl_hme](https://github.com/revidee/CoMER/tree/feature/ssl_hme), no helpers (visualization)
- [feature/ssl_helpers](https://github.com/revidee/CoMER/tree/feature/ssl_helpers), no HME100K support

## Project structure
```bash
├── README.md
├── comer               # model definition folder
├── convert2symLG       # official tool to convert latex to symLG format
├── lgeval              # official tool to compare symLGs in two folder
├── config.yaml         # config for CoMER hyperparameter
├── data.zip
├── eval_all.sh         # script to evaluate model on all CROHME test sets
├── example
│   ├── UN19_1041_em_595.bmp
│   └── example.ipynb   # HMER demo
├── lightning_logs      # training logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch=151-step=57151-val_ExpRate=0.6365.ckpt
│       ├── config.yaml
│       └── hparams.yaml
├── requirements.txt
├── scripts             # evaluation scripts
├── setup.cfg
├── setup.py
└── train.py
```

## Install dependencies   
```bash
cd CoMER
# install project 
# python >= 3.7 required. Tested with 3.7 & 3.10
conda create -y -n CoMER python=3.7
conda activate CoMER
# install pytorch >= 1.8 & torchvision >= 0.2 with cudatoolkit / rocm.
conda install pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -e .
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge

 ```

## Training
Next, navigate to CoMER folder and run `train.py`. It may take **7~8** hours on **4** NVIDIA 2080Ti gpus using ddp.
```bash
# train CoMER(Fusion) model using 2 gpus and ddp
python train.py -c config.yaml fit
```

You may change the `config.yaml` file to train different models
```yaml
# train BTTR(baseline) model
cross_coverage: false
self_coverage: false

# train CoMER(Self) model
cross_coverage: false
self_coverage: true

# train CoMER(Cross) model
cross_coverage: true
self_coverage: false

# train CoMER(Fusion) model
cross_coverage: true
self_coverage: true
```

For _single_ `gpu` usage, you may edit the `config.yaml`:
```yaml
accelerator: 'gpu'
devices: 0
```

For _single_ `cpu` user, you may edit the `config.yaml`:
```yaml
accelerator: 'cpu'
# devices: 0
```

## Evaluation
Metrics used in validation during the training process is not accurate.

For accurate metrics reported in the paper, please use tools officially provided by CROHME 2019 organizer:

A trained CoMER(Fusion) weight checkpoint has been saved in `lightning_logs/version_0`



```bash
perl --version  # make sure you have installed perl 5

unzip -q data.zip

# evaluation
# evaluate model in lightning_logs/version_0 on all CROHME test sets
# results will be printed in the screen and saved to lightning_logs/version_0 folder
bash eval_all.sh 0
```

<div align="center">    
 
# CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition  
 
[![arXiv](https://img.shields.io/badge/arXiv-2207.04410-b31b1b.svg)](https://arxiv.org/abs/2207.04410)

</div>

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
pip install "numpy<1.30" "torch==1.12.1" "torchvision==0.13.1" --index-url https://download.pytorch.org/whl/cu116
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

# evaluate given model on all CROHME test sets
# results will be printed and put into ./eval_out
bash eval_all.sh -cp my_trained_model.ckpt

# All CLI Options:
#   -cp,--cp-path <file-path: string> - Path to the checkpoint
#   -o,--out-dir <dir-path: string> (optional, def: ./eval_out) - Directory in which all results will be copied to
#   -d,--data-dir: <dir-path: string> (optional, def: ./data) - Data directory of the unzipped data.zip
#   -gpu,--gpu: <idx: int> (optional, def: 0) - index of the cuda device to use
#   -p,--pandoc: <file-path: string> (optional, def: pandoc) - path to the pandoc executable, if it needs to be overwritten
```
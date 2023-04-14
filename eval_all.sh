#!/bin/bash

# USAGE: eval_all.sh
#   -cp,--cp-path <file-path: string> - Path to the checkpoint
#   [-o,--out-dir <dir-path: string>] (optional, def: ./eval_out) - Directory in which all results will be copied to
#   [-d,--data-dir: <dir-path: string>] (optional, def: ./data) - Data directory of the unzipped data.zip
#   [-gpu,--gpu: <idx: int>] (optional, def: 0) - index of the cuda device to use
#   [-p,--pandoc: <file-path: string>] (optional, def: pandoc) - path to the pandoc executable, if it needs to be overwritten
#   [-aug,--augmentation: <aug_mode: string>] (optional, def: '') - augmentation to apply to the test set, defaults to none
#   [-m,--model: <model_name: string>] (optional, def: sup) - the model instance to load. Available models are listed in model_lookups.py.
# Example:
# ./eval_all.sh -cp ./lightning_logs/version_14/checkpoints/epoch=209-step=238770-val_ExpRate=0.6230.ckpt -o ./eval/version_14 -d /data -gpu 1
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -cp|--cp-path)
      checkpoint_path="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--out-dir)
      out_dir=$(echo $2 | sed 's:/*$::')
      shift # past argument
      shift # past value
      ;;
    -d|--data-dir)
      data_dir="$2"
      shift # past argument
      shift # past value
      ;;
    -gpu|--gpu)
      gpu="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--seed)
      seed="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--pandoc)
      pandoc_path="$2"
      shift # past argument
      shift # past value
      ;;
    -aug|--augmentation)
      aug="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--model)
      model="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters


if [ -z "$out_dir" ]; then
  out_dir='./eval_out'
fi
mkdir -p $out_dir

if [ -z "$data_dir" ]; then
  data_dir='./data'
fi

if [ -z "$gpu" ]; then
  gpu='0'
fi

if [ -z "$seed" ]; then
  seed='7'
fi

if [ -z "$aug" ]; then
  aug=""
else
  aug="-aug "$aug""
fi

if [ -z "$model" ]; then
  model=""
else
  model="-m "$model""
fi

single_num_regex='^[0-9]$'
if ! [[ $gpu =~ $single_num_regex ]] ; then
   echo "fatal: given gpu index is not a single number (expected a device id, eg for cuda:0, \"0\")." >&2;
   echo "given: ${gpu}"
   exit 1
fi

# set PANDOC_EXEC env variable, which is being used by tex2mml / tex2symlg.
# Used, when pandoc is not installed globally
if [ -z "$pandoc_path" ]; then
  export PANDOC_EXEC='pandoc'
else
  export PANDOC_EXEC="$pandoc_path"
fi

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
# install lgeval and tex2symlg (in CoMER folder)
export LgEvalDir=$parent_path/lgeval
export Convert2SymLGDir=$parent_path/convert2symLG
export PATH=$PATH:$LgEvalDir/bin:$Convert2SymLGDir

for year in '2014' '2016' '2019'
do
    echo '****************' start evaluating CROHME $year '****************'
    bash $parent_path/scripts/test/eval.sh -cp $checkpoint_path -o $out_dir -d $data_dir -y $year -gpu $gpu -s $seed $aug $model
    echo
done


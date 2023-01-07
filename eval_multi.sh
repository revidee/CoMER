#!/bin/bash

# USAGE: eval_multi.sh
# Runs eval_all with multiple seeds. Useful when evaluating with random augmentations.
#   -cp,--cp-path <file-path: string> - Path to the checkpoint
#   [-o,--out-dir <dir-path: string>] (optional, def: ./eval_out) - Directory in which all results will be copied to
#   [-d,--data-dir: <dir-path: string>] (optional, def: ./data) - Data directory of the unzipped data.zip
#   [-gpu,--gpu: <idx: int>] (optional, def: 0) - index of the cuda device to use
#   [-p,--pandoc: <file-path: string>] (optional, def: pandoc) - path to the pandoc executable, if it needs to be overwritten

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
      data_dir=$(echo $2 | sed 's:/*$::')
      shift # past argument
      shift # past value
      ;;
    -gpu|--gpu)
      gpu="$2"
      shift # past argument
      shift # past value
      ;;
    -aug|--augmentation)
      aug="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--pandoc)
      pandoc_path="$2"
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

if [ -z "$data_dir" ]; then
  data_dir='./data'
fi

if [ -z "$gpu" ]; then
  gpu='0'
fi

if [ -z "$aug" ]; then
  aug=""
else
  aug="-aug "$aug""
fi

if [ -z "$pandoc_path" ]; then
  pandoc_path=""
else
  pandoc_path="-p "$pandoc_path""
fi

single_num_regex='^[0-9]$'
if ! [[ $gpu =~ $single_num_regex ]] ; then
   echo "fatal: given gpu index is not a single number (expected a device id, eg for cuda:0, \"0\")." >&2;
   echo "given: ${gpu}"
   exit 1
fi

for seed in '1' '2' '3' '7' '42' '8'; do
    bash ./eval_all.sh -cp $checkpoint_path -o "$out_dir"_$seed -d $data_dir -gpu $gpu -s $seed $aug $pandoc_path
done



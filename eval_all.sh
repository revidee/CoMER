#!/bin/bash

# USAGE: eval_all.sh
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
      data_dir="$2"
      shift # past argument
      shift # past value
      ;;
    -gpu|--gpu)
      gpu="$2"
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

single_num_regex='^[0-9]$'
if ! [[ $gpu =~ $single_num_regex ]] ; then
   echo "fatal: given gpu index is not a single number (expected a device id, eg for cuda:0, \"0\")." >&2;
   echo "given: ${gpu}"
   exit 1
fi

# set PANDOC_PATH env variable, which is being used by tex2mml / tex2symlg.
# Used, when pandoc is not installed globally
if [ -z "$pandoc_path" ]; then
  export PANDOC_PATH='pandoc'
else
  export PANDOC_PATH="$pandoc_path"
fi


# install lgeval and tex2symlg (in CoMER folder)
export LgEvalDir=$(pwd)/lgeval
export Convert2SymLGDir=$(pwd)/convert2symLG
export PATH=$PATH:$LgEvalDir/bin:$Convert2SymLGDir

for year in '2014' '2016' '2019'
do
    echo '****************' start evaluating CROHME $year '****************'
    bash scripts/test/eval.sh -cp $checkpoint_path -o $out_dir -d $data_dir -y $year -gpu $gpu
    echo
done

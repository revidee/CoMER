#!/bin/bash
# USAGE: ./scripts/test/eval.sh
#   -cp,--cp-path <file-path: string> - Path to the checkpoint
#   -y,--year <2014/2016/2019>
#   [-o,--out-dir <dir-path: string>] (optional, def: ./eval_out) - Directory in which all results will be copied to
#   [-d,--data-dir: <dir-path: string>] (optional, def: ./data) - Data directory of the unzipped data.zip
#   [-gpu,--gpu: <idx: int>] (optional, def: 0) - index of the cuda device to use





POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -cp|--cp-path)
      chk_point_path="$2"
      shift # past argument
      shift # past value
      ;;
    -y|--year)
      test_year="$2"
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
    -aug|--augmentation)
        aug="$2"
        shift # past argument
        shift # past value
        ;;
    -s|--seed)
      seed="$2"
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

if [ -z "$chk_point_path" ]; then
  echo "fatal: checkpoint path not given";
  exit 1
fi

if [ -z "$test_year" ]; then
  echo "fatal: year not given";
  exit 1
fi

if [ -z "$out_dir" ]; then
  out_dir='./eval_out'
fi

if [ -z "$aug" ]; then
  aug=""
else
  aug="--aug "$aug""
fi

if [ -z "$data_dir" ]; then
  data_dir='./data'
fi

if [ -z "$gpu" ]; then
  gpu='0'
fi

if ! [[ $gpu =~ $single_num_regex ]] ; then
   echo "fatal: given gpu index is not a single number (expected a device id, eg for cuda:0, \"0\")." >&2;
   echo "given: ${gpu}"
   exit 1
fi

data_dir=$(readlink -f $data_dir)
out_dir=$(readlink -f $out_dir)
chk_point_path=$(readlink -f $chk_point_path)

# clean out
rm -rf $out_dir/test_temp/
rm -rf $out_dir/Results_pred_symlg/

# generate predictions
python -m scripts.test.test $chk_point_path --year $test_year --gpu $gpu --seed $seed $aug

mkdir -p $out_dir/test_temp/
mv result$gpu.zip $out_dir/$test_year.zip
mv stats$gpu.txt $out_dir/"$test_year"_stats.txt
unzip -q $out_dir/$test_year.zip -d $out_dir/test_temp/result

dir=$(pwd)
cd $out_dir

# convert tex to symlg
tex2symlg $out_dir/test_temp/result $out_dir/test_temp/pred_symlg

# evaluate two symlg folder
evaluate $out_dir/test_temp/pred_symlg $data_dir/$test_year/symLg >/dev/null 2>&1

# copy results
cp ./Results_pred_symlg/Summary.txt $out_dir/"$test_year"_Summary.txt
cp ./Results_pred_symlg/FileMetrics.csv $out_dir/"$test_year"_FileMetrics.csv

abs_results=$(readlink -f ./Results_pred_symlg/Summary.txt)

cd $dir

# extract evaluation result and save to target folder
python -m scripts.test.extract_exprate 4 --path $abs_results >&1 | tee $out_dir/$test_year.txt

rm -rf $out_dir/Results_pred_symlg
rm -rf $out_dir/test_temp/



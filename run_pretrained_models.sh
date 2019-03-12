#!/usr/bin/env bash

#set -x
#sh run_pretrained_models.sh quasart /mnt/nfs/scratch1/rajarshi/data/ICLR_code_release/data/ models/ /mnt/nfs/scratch1/rajarshi/
if [ "$#" -ne 4 ]; then
    echo "Usage: /bin/bash run_pretrained_models.sh dataset_name data_dir model_dir out_dir"
    echo "dataset_name -- one of triviaqa|searchqa|quasart"
    echo "data_dir -- top level dir path to the downloaded and unzipped data dir"
    echo "model_dir -- top level dir path to the downloaded and unzipped pretrained model dir"
    echo "out_dir -- a directory to write logs"
    exit 1
fi
dataset_name=$1
data_dir=$2
model_dir=$3
out_dir=$4

echo "Evaluating for $dataset_name..."

if [ $dataset_name = "triviaqa" ]; then
       num_paras_test=10
       multi_step_reasoning_steps=3
       test_batch_size=10
elif [ $dataset_name = "searchqa" ]; then
       num_paras_test=10
       multi_step_reasoning_steps=7
       test_batch_size=32
elif [ $dataset_name = "quasart" ]; then
       num_paras_test=25
       multi_step_reasoning_steps=5
       test_batch_size=32
fi

python scripts/reader/train.py --domain web-open --num_paras_test $num_paras_test \
--multi_step_reasoning_steps $multi_step_reasoning_steps \
--dataset_name $dataset_name --top-spans 10 --eval_only 1 \
--pretrained $model_dir/$dataset_name/model.mdl \
--model_dir $out_dir \
--data_dir $data_dir \
--test_batch_size $test_batch_size \
--saved_para_vector $data_dir/$dataset_name/paragraph_vectors/web-open/

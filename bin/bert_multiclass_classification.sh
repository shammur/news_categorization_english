#! /bin/bash

cd bin/transformers/examples
source $PATH_TO_ENV/news_cat_bert_env/bin/activate

export HOME_DIR="$PWD"
export TASK_NAME=multiclass
model=bert-base-uncased
outputDir=$HOME_DIR/output_multi_class_${model}
export data_dir=$HOME_DIR"/data_bert_model/"
mkdir -p $data_dir

python run_glue_multiclass.py --model_type bert --model_name_or_path bert-base-uncased --task_name $TASK_NAME --do_test --do_lower_case --data_dir $data_dir \
--max_seq_length 128 --per_gpu_eval_batch_size=32  --per_gpu_train_batch_size=32   --learning_rate 5e-5 --num_train_epochs 20.0 \
--test_file $HOME_DIR/data/sample_data.csv  --output_dir $outputDir --results_file all_data_test_results.txt


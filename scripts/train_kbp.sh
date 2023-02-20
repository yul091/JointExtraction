# Training
MODEL_PATH="gpt2-medium"
TRAIN_FILE="examples/test_data/KBP_train_cluster.json" # 144646
VAL_FILE="examples/test_data/KBP_test_cluster.json" # 919
# OUTPUT_DIR="examples/tok_cls_result/KBP_baseline_avgpool"
OUTPUT_DIR="examples/tok_cls_result/KBP_bootstrap_maxpool"


CUDA_VISIBLE_DEVICES=3 python examples/token-classification-gpt2/run_jointmodel_kbp.py \
--model_name_or_path $MODEL_PATH --classifier_type "crf" \
--train_file $TRAIN_FILE --validation_file $VAL_FILE \
--output_dir $OUTPUT_DIR --do_eval --do_train \
--evaluation_strategy epoch --load_best_model_at_end \
--metric_for_best_model eval_f1 --greater_is_better True \
--per_device_train_batch_size 5 --per_device_eval_batch_size 20 \
--gradient_accumulation_steps 16 --overwrite_cache \
--use_negative_sampling --num_train_epochs 100 \
--beta 1.0 --sample_rate 0.1 --boot_start_epoch 5 --threshold 0.7 

# --baseline
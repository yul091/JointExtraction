# Training
MODEL_PATH='lstm'
TRAIN_FILE="examples/test_data/new_joint_train_NYT_1over4.json" # 172718
# TRAIN_FILE="examples/test_data/train_NYT_1over4_preprocessed.json" # 200446
VAL_FILE="examples/test_data/new_joint_test_part_NYT.json" # 1680
OUTPUT_DIR="examples/tok_cls_result/NYT_lstm_noattbase"


CUDA_VISIBLE_DEVICES=2 python examples/token-classification-lstm/run_jointmodel.py \
--model_name_or_path $MODEL_PATH --classifier_type "crf" \
--train_file $TRAIN_FILE --validation_file $VAL_FILE \
--output_dir $OUTPUT_DIR --do_eval --do_train \
--evaluation_strategy epoch --load_best_model_at_end \
--metric_for_best_model eval_f1 --greater_is_better True \
--per_device_train_batch_size 64 --per_device_eval_batch_size 64 \
--gradient_accumulation_steps 16 --overwrite_cache \
--use_negative_sampling --sample_rate 0.1 --num_train_epochs 100 \
--beta 0 --alpha 0 --boot_start_epoch 10 --threshold 0.5 --baseline

# # Evaluation
# TRAIN_FILE="examples/test_data/new_joint_train_NYT_1over4.json" # 170000
# # TRAIN_FILE="examples/test_data/joint_train_NYT(relation).json" # 160000
# # VAL_FILE="examples/test_data/new_joint_test_part_NYT.json"
# # VAL_FILE="examples/test_data/joint_test_NYT(entity).json"
# VAL_FILE="examples/test_data/joint_test_NYT(relation).json"
# # MODEL_PATH="examples/tok_cls_result/NYT_baseline_maxpool/checkpoint-49854" # (1/4 NYT baseline maxpool)
# # MODEL_PATH="examples/tok_cls_result/NYT_baseline_avgpool/checkpoint-49854" # (1/4 NYT baseline avgpool)
# # MODEL_PATH="examples/tok_cls_result/NYT_baseline_noattreg/checkpoint-59350" # (1/4 NYT bootstrap noattreg)
# # MODEL_PATH="examples/tok_cls_result/NYT_bootstrap_maxpool/checkpoint-82119" # (1/4 NYT bootstrap maxpool)
# # MODEL_PATH="examples/tok_cls_result/NYT_bootstrap_avgpool/checkpoint-169734" # (1/4 NYT bootstrap avgpool)
# MODEL_PATH="examples/tok_cls_result/NYT_bootstrap_avgpool_merge/checkpoint-182966" # (1/4 NYT bootstrap avgpool merge)
# OUTPUT_DIR="examples/tok_cls_result/train_quarter"

# CUDA_VISIBLE_DEVICES=2 python examples/token-classification-gpt2/run_jointmodel.py \
# --model_name_or_path $MODEL_PATH --classifier_type "crf" \
# --train_file $TRAIN_FILE --output_dir $OUTPUT_DIR \
# --validation_file $VAL_FILE --do_eval --overwrite_cache \
# --evaluation_strategy epoch --load_best_model_at_end \
# --metric_for_best_model eval_f1 --greater_is_better True \
# --per_device_eval_batch_size 20 --gradient_accumulation_steps 16  

# --max_ent_range 3 --boot_start_epoch 5 --threshold 0.5
# --use_subtoken_mask --baseline --boot_start_epoch 3
# --max_train_samples 1000 --max_val_samples 50 --max_new_patterns 10

# Training
MODEL_PATH="bert-large-uncased"
# MODEL_PATH="examples/tok_cls_result/NYT_BERT_bootstrap_avgpool_merge/checkpoint-7377" 
# TRAIN_FILE="examples/test_data/joint_train_NYT(relation).json" # 160620
TRAIN_FILE="examples/test_data/new_joint_train_NYT_1over4.json" # 172718
VAL_FILE="examples/test_data/new_joint_test_part_NYT.json" # 1680
# TRAIN_FILE="examples/test_data/train_KBP.json" # 144646
# VAL_FILE="examples/test_data/test_KBP.json" # 1680
OUTPUT_DIR="examples/tok_cls_result/NYT_bootstrap_BERT_uncased_new"


CUDA_VISIBLE_DEVICES=0 python examples/token-classification-bert/run_jointmodel.py \
--model_name_or_path $MODEL_PATH --classifier_type "crf" \
--train_file $TRAIN_FILE --validation_file $VAL_FILE \
--output_dir $OUTPUT_DIR --do_train --do_eval \
--evaluation_strategy epoch --load_best_model_at_end \
--metric_for_best_model eval_f1 --greater_is_better True \
--per_device_train_batch_size 5 --per_device_eval_batch_size 20 \
--gradient_accumulation_steps 16 --overwrite_cache \
--use_negative_sampling --num_train_epochs 100 \
--beta 1.0 --sample_rate 0.1 --boot_start_epoch 5 --threshold 0.5 


# --max_ent_range 3 --boot_start_epoch 5 --threshold 0.5
# --use_subtoken_mask --baseline --boot_start_epoch 3
# --max_train_samples 1000 --max_val_samples 50 --max_new_patterns 10

python run_new.py \
    --output_dir=./saved_models \
    --model_type=deepseek \
    --tokenizer_name=/home/rsr200002/LLMs/deepseek_coder_67b_instruct \
    --model_name_or_path=/home/rsr200002/LLMs/deepseek_coder_67b_instruct \
    --do_train \
    --train_data_file=../dataset/dev.jsonl \
    --eval_data_file=../dataset/dev.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 4 \
    --project vul \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 12 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
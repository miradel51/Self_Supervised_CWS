export CUDA_VISIBLE_DEVICES=1

python run_mlm_scratch.py \
  --model_name_or_path=bert-base-chinese \
  --train_data_file=./your_dataset_dir/combined.shuf.train.cut \
  --eval_data_file=./your_dataset_dir/combined.shuf.dev.cut \
  --output_dir=./your_model_dir/mlm_scratch_new_05_08_other_mkc_2_cut \
  --block_size=384 \
  --num_train_epochs=10 \
  --per_gpu_train_batch_size=32 \
  --learning_rate=1e-4 \
  --weight_decay=0.01 \
  --save_steps=500 \
  --seed=42 \
  --do_train \
  --do_eval \
  --main_metric=eval_loss \
  --metric_type=-1 \
  --mask_count=2

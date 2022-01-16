export CUDA_VISIBLE_DEVICES=0

python ./pred_mlm_mrt.py \
  --model_name_or_path=./your_model_dir/mlm_scratch_new_05_08_other_mkc_2_cut \
  --eval_data_file=./your_dataset_dir/pku/train.txt.char \
  --output_dir=./your_dataset_dir/pku/pred_mlm.txt \
  --block_size=384 \
  --do_predict \
  --lm_format_pred_result \
  --mask_count=2

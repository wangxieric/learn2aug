export CUDA_VISIBLE_DEVICES=0

root_path = ''

# train
python3 train_simpletod.py \
--model_save_name=model1_rand \
--output_path="${root_path}outputs/" \
--input="${root_path}todkg_dataset/runs/model1/model1.lm.rand.input.train_final.txt" \
--dev_input="${root_path}todkg_dataset/runs/model1/model1.lm.rand.input.dev_final.txt" \
--eos_token_id=50256 \
--batch_size=16 \
--max_epoch=50 \
--learning_rate=1e-4 \
--report_loss=100 \
--report=500 \
--max_seq_len=512 \
--neg_sample \
--neg_sample_rate=3
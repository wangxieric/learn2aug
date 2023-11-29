export CUDA_VISIBLE_DEVICES=1

root_path="/home/xwang/learn2aug/data/"

# # train
python3 train_simpletod.py \
--model_save_name=model1_rand \
--output_path="${root_path}SimpleTOD/outputs/" \
--input="${root_path}SimpleTOD/processed/model1.lm.rand.input.train_final.txt" \
--dev_input="${root_path}SimpleTOD/processed/model1.lm.rand.input.dev_final.txt" \
--eos_token_id=50256 \
--batch_size=8 \
--max_epoch=50 \
--learning_rate=1e-4 \
--report_loss=100 \
--report=500 \
--max_seq_len=512 \
--neg_sample \
--neg_sample_rate=3
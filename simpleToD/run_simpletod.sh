export CUDA_VISIBLE_DEVICES=0

root_path="/home/xwang/learn2aug/data/"

# # train
# python3 train_simpletod.py \
# --model_save_name=model1_rand \
# --output_path="${root_path}SimpleTOD/outputs/" \
# --input="${root_path}SimpleTOD/processed/model1.lm.rand.input.train_final.txt" \
# --dev_input="${root_path}SimpleTOD/processed/model1.lm.rand.input.dev_final.txt" \
# --eos_token_id=50256 \
# --batch_size=8 \
# --max_epoch=50 \
# --learning_rate=1e-4 \
# --report_loss=100 \
# --report=500 \
# --max_seq_len=512 \
# --neg_sample \
# --neg_sample_rate=3

# # test retrieved
python test_simpletod.py \
--saved_model_path="model1_rand_20231129213354/saved_model/loads/18/model.pt" --output_path="${root_path}SimpleTOD/outputs/" \
--model_dir_name="model1_gold_action_retrieved_kg_gold_decision" \
--test_input="${root_path}SimpleTOD/processed/model1.lm.input.eval.test_retrieved.txt" \
--test_input_gold_action="${root_path}SimpleTOD/processed/model1.lm.input.eval.goldaction.test_retrieved.txt" \
--test_input_gold_kg="${root_path}SimpleTOD/processed/model1.lm.input.eval.goldkg.test_retrieved.txt" \
--test_input_gold_decision="${root_path}SimpleTOD/processed/model1.lm.input.eval.golddecision.test_retrieved.txt" \
--test_oracle_input="${root_path}SimpleTOD/processed/model1.lm.input.test_retrieved.txt" \
--test_input_original="${root_path}SimpleTOD/processed/processed_model1_test_retrieved.json" \
--test_inter="${root_path}SimpleTOD/processed/test_retrieved_inter.json" \
--test_inter_res="${root_path}SimpleTOD/processed/test_retrieved_inter.json" \
--en_schema="${root_path}SimpleTOD/entity_schemas/schema_all.json" \
--num_passages=2 \
--num_para=2 \
--eos_token_id=50256 \
--batch_size=1 \
--max_seq_len=1024 \
--gold_action \

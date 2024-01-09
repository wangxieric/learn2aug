import json

generated_outputs_dir = '/home/xwang/learn2aug/data/SimpleTOD/outputs/inference_only_model1_gold_action_retrieved_kg_gold_decision/results/result_final_noisy_knowledge-updated[test].json'
generated_outputs = json.load(open(generated_outputs_dir, 'r'))

logits = []

for gen in generated_outputs:
    logits.append(gen.split(' ')[-1])


# save logits
with open('/home/xwang/learn2aug/data/SimpleTOD/outputs/inference_only_model1_gold_action_retrieved_kg_gold_decision/results/noisy_knowledge-updated_logits.txt', 'w') as f:
    json.dump(logits, f)

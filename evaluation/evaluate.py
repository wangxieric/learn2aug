import gem_metrics

pred_dir = '/home/xwang/learn2aug/data/SimpleTOD/outputs/inference_only_model1_gold_action_retrieved_kg_gold_decision/results/reformatted_result_final_gold_knowledge_retrieved-updated.txt'
ref_dir = '/home/xwang/learn2aug/data/KETOD/full_ketod/test_system_utterances.txt'

with open(pred_dir, 'r') as f:
    list_of_predictions = f.readlines()

with open(ref_dir, 'r') as f:
    list_of_references = f.readlines()

preds = gem_metrics.texts.Predictions(list_of_predictions)
refs = gem_metrics.texts.References(list_of_references)

result = gem_metrics.compute(preds, refs, metrics_list=['bleu', 'rouge', 'bertscore'])  # add list of desired metrics here
print(result)
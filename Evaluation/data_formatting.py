# reformat the string list with taggers to a list of utterance only

import json

'''
    Prepare the gold truth for comparison
'''

test_data_dir = '/home/xwang/learn2aug/data/KETOD/full_ketod/test.json'
test_data = json.load(open(test_data_dir, 'r'))

extracted_gold = []
for data in test_data:
    for turn in data['turns']:
        if turn['speaker'] == 'SYSTEM':
            extracted_gold.append(turn['utterance'])

with open('/home/xwang/learn2aug/data/KETOD/full_ketod/test_system_utterances.txt', 'w') as f:
    for sentence in extracted_gold:
        f.write("{0}\n".format(sentence))

'''
    Reformat the generated utterances for evaluation
'''

generated_outputs_dir = '/home/xwang/learn2aug/data/SimpleTOD/outputs/inference_only_model1_gold_action_retrieved_kg_gold_decision/results/result_final.json'
generated_outputs = json.load(open(generated_outputs_dir, 'r'))

reformatted_generations = []
for gen in generated_outputs:
    reformatted_generations.append(gen.split('<|response|>')[1].split('<|endofresponse|>')[0])

with open('/home/xwang/learn2aug/data/SimpleTOD/outputs/inference_only_model1_gold_action_retrieved_kg_gold_decision/results/reformatted_result_final.txt', 'w') as f:
    for sentence in reformatted_generations:
            f.write("{0}\n".format(sentence))

print(len(reformatted_generations))
import json
from datasets import Dataset, DatasetDict

def join_training_samples(data_dir_list, output_dir):
    "join multiple training samples into one dataset"
    for data_dir in data_dir_list:
        data = json.load(open(data_dir, 'r'))
        if data_dir == data_dir_list[0]:
            output_data = data
        else:
            output_data.extend(data)
    print("dataset size: ", len(output_data))
    json.dump(output_data, open(output_dir, 'w'))



def prepare_dataset(dataset_dir, output_dir):
    "create a dataset as instruction input to fine tune a gating model"

    input_data = json.load(open(dataset_dir, 'r'))
    inputs = []
    targets = []
    instructions = []
    #  user context as input and logits as target
    for row in input_data:
        row = row.split()
        context = row[:-2]
        logits = row[-1]
        instruction = "Estimate the lowest logit of generating a response to the input text. \
                        The estimation examines the previous utterances, estimated attributes \
                        and especially the contribution of the augmented knowledge."
        instructions.append(instruction)
        inputs.append(' '.join(context))
        targets.append(logits)
    
    data_dict = {
        "train" : {
            "instruction": instructions,
            "text": inputs,
            "target": targets
        }
    }

    dataset = DatasetDict()
    for k, v in data_dict.items():
        dataset[k] = Dataset.from_dict(v)

    dataset.save_to_disk(output_dir)


"""
    Prepare the training data for fine-tune a llama-based gating model
"""
# This dataset combines various strategies in combining the knowledge, including the use of bert retrieved knowledge, noisy knolwedge, gold knowledge, and no knowledge
data_dir = '/home/xwang/learn2aug/data/SimpleTOD/outputs/inference_only_model1_gold_action_retrieved_kg_gold_decision/results/'
data_dir_list = [
    data_dir + 'result_final_knowledge[test].json',
    data_dir + 'result_final_noisy_knowledge-updated[test].json',
    data_dir + 'result_final_gold_knowledge_retrieved-updated[test].json',
    data_dir + 'result_final_gold_knowledge[test].json',
    data_dir + 'result_final_no_knowledge[test].json'
]
output_dir = data_dir + 'combined_input_for_gating_model.json'
join_training_samples(data_dir_list, output_dir)

resulting_dataset = data_dir +  'dataset_for_gating_model'
prepare_dataset(output_dir, resulting_dataset)


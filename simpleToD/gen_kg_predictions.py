# Copyright (c) Meta Platforms, Inc. and its affiliates.
import json
from tqdm import tqdm


from utils import get_kg_snippets_dict
'''
merge kg selection results with original
'''


def merge_train(json_in, json_out, topn_snippets=3, is_test=True):
    '''
    merge train or dev
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    count_missing_snippets = 0
    count_total_snippets = 0
    for each_data in tqdm(data_all):

        all_kg_snippets_dict = get_kg_snippets_dict(each_data)
        # print(all_kg_snippets_dict.keys())
        for turn in each_data["turns"]:
            if turn["speaker"] == "SYSTEM":

                this_retrieved_kg_text = []
                if turn["enrich"]:
                    if not is_test:
                        this_retrieved_kg_text = turn["kg_snippets_text"][:]

                        retrieved_ind = 0
                        while len(this_retrieved_kg_text) < topn_snippets and retrieved_ind < len(turn["retrieved"]):
                            # make up with the retrieved ones
                            this_added_ind = turn["retrieved"][retrieved_ind]
                            if this_added_ind not in turn["kg_snippets"]:
                                added_kg_text = all_kg_snippets_dict[turn["retrieved"][retrieved_ind]] if turn["retrieved"][retrieved_ind] in all_kg_snippets_dict.keys() else "no snippet"
                                if added_kg_text == "no snippet":
                                    count_missing_snippets += 1
                                count_total_snippets += 1
                                this_retrieved_kg_text.append(added_kg_text)
                            retrieved_ind += 1
                    else:
                        for each_added in turn["retrieved"]:
                            if each_added not in all_kg_snippets_dict.keys():
                                count_missing_snippets += 1
                            count_total_snippets += 1
                            this_retrieved_kg_text.append(all_kg_snippets_dict[each_added] if each_added in all_kg_snippets_dict.keys() else "no snippet")

                else:
                    # print("turn:", turn["retrieved"])
                    for each_added in turn["retrieved"]:
                        if each_added not in all_kg_snippets_dict.keys():
                                count_missing_snippets += 1
                        count_total_snippets += 1
                        this_retrieved_kg_text.append(all_kg_snippets_dict[each_added] if each_added in all_kg_snippets_dict.keys() else "no snippet")

                turn["merge_retrieved"] = this_retrieved_kg_text

    print("count_missing_snippets:", count_missing_snippets)
    print("count_total_snippets:", count_total_snippets)
    with open(json_out, "w") as f:
        json.dump(data_all, f, indent=4)




root = "../data/outputs/"
tgt = "../data/todkg_dataset/runs/"

# train
json_in = root + "inference_only_20231128232324_kg_select_bert_base_model2_new/results/test/predictions.json"
json_out = tgt + "model1/" + "train_final.json"

merge_train(json_in, json_out, topn_snippets=3, is_test=False)

# # dev
# json_in = root + "inference_only_20231128165013_kg_select_bert_base_valid_2/results/test/predictions.json"
# json_out = tgt + "model1/" + "dev_final.json"

# merge_train(json_in, json_out, topn_snippets=3, is_test=False)


# test
# json_in = root + "inference_only_20231128114114_kg_select_bert_base_test/results/test/predictions.json"
# json_out = tgt + "model1/" + "test_retrieved.json"

# merge_train(json_in, json_out, topn_snippets=3, is_test=True)
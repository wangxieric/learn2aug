'''
    Author: Xi Wang
    Date: Nov. 2023
    Description: generate full KETOD dataset by augmenting SGD with labelled data from KETOD
'''
import json
import os


def gen_ketod(json_in, sgd_folder_in, json_out, mode="train"):
    with open(json_in) as f_in:
        data = json.load(f_in)
    
    mode_sgd_train = {}
    sgd_folder = os.path.join(sgd_folder_in, "train")
    for file in os.listdir(sgd_folder):
        if "dialogues" in file:
            with open(os.path.join(sgd_folder, file)) as f:
                sgd_data = json.load(f)
                for each_data in sgd_data:
                    assert each_data["dialogue_id"] not in mode_sgd_train
                    mode_sgd_train[each_data["dialogue_id"]] = each_data

    mode_sgd = {}
    if mode == "train":
        mode_sgd = mode_sgd_train
    else:
        sgd_folder = os.path.join(sgd_folder_in, mode)
        for filename in os.listdir(sgd_folder):
            if "dialogues" in filename:
                with open(os.path.join(sgd_folder, filename)) as f:
                    sgd_data = json.load(f)
                    for each_data in sgd_data:
                        assert each_data["dialogue_id"] not in mode_sgd
                        mode_sgd[each_data["dialogue_id"]] = each_data

    for each_data in data:
        this_sgd = mode_sgd[each_data["dialogue_id"]] if each_data["dialogue_id"] in mode_sgd \
                    else mode_sgd_train[each_data["dialogue_id"]]
        this_final_turns = []
        for sgd_turn, ketod_turn in zip(this_sgd["turns"], each_data["turns"]):
            # unify both dictionaries of dialogues
            final_turn = sgd_turn | ketod_turn
            this_final_turns.append(final_turn)

        each_data["turns"] = this_final_turns

    print(mode + " data size: ", len(data))

    with open(json_out, "w") as f_out:
        json.dump(data, f_out, indent=4)


if __name__ == "__main__":
    root = "data/"
    ketod_release = root + "KETOD/"

    ketod_release_train = ketod_release + "train_ketod.json"
    ketod_release_dev = ketod_release + "dev_ketod.json"
    ketod_release_test = ketod_release + "test_ketod.json"

    sgd = root + "SGD/"
    target_folder = root + "Full_KETOD/"
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    full_train = target_folder + "train.json"
    full_dev = target_folder + "dev.json"
    full_test = target_folder + "test.json"

    # augment SGD with labelled data from KETOD
    gen_ketod(ketod_release_train, sgd, full_train, mode="train")
    gen_ketod(ketod_release_dev, sgd, full_dev, mode="dev")
    gen_ketod(ketod_release_test, sgd, full_test, mode="test")


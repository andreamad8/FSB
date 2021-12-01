import json
import glob
import os
from collections import defaultdict
import tabulate
import numpy as np

file_list = []
for file in glob.glob("../generations/babi5-first*"):
    if "_5_" in file:
        file_temp = file.replace("first", "second").replace("_5_", "_2_")
    elif "_8_" in file:
        file_temp = file.replace("first", "second").replace("_8_", "_2_")
    file_list.append([file, file_temp])

table = []
for first_file, second_file in file_list:
    # check if the files exist
    if not os.path.isfile(first_file):
        print("File {} does not exist".format(first_file))
        continue
    if not os.path.isfile(second_file):
        print("File {} does not exist".format(second_file))
        continue

    # load the files
    with open(first_file, "r") as f:
        first_data = json.load(f)
    with open(second_file, "r") as f:
        second_data = json.load(f)
    
    # merge dialogue with same id
    merged_data = defaultdict(list)
    for d in first_data["generation"]:
        merged_data[d["id"]].append(d)
    for d in second_data["generation"]:
        merged_data[d["id"]].append(d)

    # create the final list of dialogue
    final_data = []
    for key, value in merged_data.items():
        final_data.append({"id": key, "dialogue": value[0]["dialogue"]+value[1]["dialogue"]})

    if "OOV" in first_file:
        # load gold file
        with open("../data/dialog-bAbI-tasks/bAbI-dial-5-OOV-test.json", "r") as f:
            gold_data = json.load(f)

    else:
        # load gold file
        with open("../data/dialog-bAbI-tasks/bAbI-dial-5-test.json", "r") as f:
            gold_data = json.load(f)

    # merge dialogue with same id in gold
    merged_gold_data = defaultdict(list)
    for d in gold_data:
        merged_gold_data[d["id"]].append(d)
    
    # create final list of gold
    final_gold_data = []
    for key, value in merged_gold_data.items():
        # print(len(value))
        # print(len(value[0]["dialogue"]))
        # print(len(value[1]["dialogue"]))
        # print(value[0]["dialogue"]+value[1]["dialogue"])
        # input()
        # print(key,len(value[0]["dialogue"]+value[1]["dialogue"]))
        final_gold_data.append({"id": key, "dialogue": value[0]["dialogue"]+value[1]["dialogue"]})

    # score final data against gold data
    acc = 0
    dialogue_acc = 0
    total = 0
    assert len(final_data) == len(final_gold_data)
    for gold_dialogue, final_dialogue in zip(final_gold_data, final_data):
        assert gold_dialogue["id"] == final_dialogue["id"]
        assert len(gold_dialogue["dialogue"]) == len(final_dialogue["dialogue"])
        temp_acc = 0
        for gold_turn, final_turn in zip(gold_dialogue["dialogue"], final_dialogue["dialogue"]):
            total += 1
            if gold_turn[1].strip() == final_turn[0].strip():
                temp_acc += 1
                acc += 1
            # else:
            #     print(gold_turn[1])
            #     print(final_turn[0])
            #     input()
        if temp_acc == len(gold_dialogue["dialogue"]):
            dialogue_acc += 1

    dataname, _, model, _, trial = first_file.split("/")[-1].replace(".json","").split("_")
    # print(first_file.replace("../generations/", "").replace("first",""))
    # print("Accuracy: {}".format(acc/total))
    # print("Dialogue Accuracy: {}".format(dialogue_acc/len(final_gold_data)))
    # print()
    table.append({"Data":dataname.replace("-first",""), "Model":model, "Trial":trial, "Accuracy":100*(acc/total), "Dialogue Accuracy": 100*(dialogue_acc/len(final_data))})

# group table by data and model
table_grouped = defaultdict(list)
for row in table:
    table_grouped[(row["Data"], row["Model"])].append(row)

# compute the mean/std of the acc and dialogue acc and put them in the table
final_table = []
for key, value in table_grouped.items():
    acc = []
    dialogue_acc = []
    for row in value:
        acc.append(row["Accuracy"])
        dialogue_acc.append(row["Dialogue Accuracy"])
    # format is (mean, std) using three 2 digits for acc and dialogue_acc
    final_table.append({"Data":key[0], "Model":key[1], "Accuracy": "{:.2f}".format(np.mean(acc))+" ({:.2f})".format(np.std(acc)), "Dialogue Accuracy": "{:.2f}".format(np.mean(dialogue_acc))+" ({:.2f})".format(np.std(dialogue_acc))})

    # final_table.append({"Data":key[0], "Model":key[1], "Accuracy":np.mean(acc), "Accuracy STD":np.std(acc), "Dialogue Accuracy":np.mean(dialogue_acc), "Dialogue Accuracy STD":np.std(dialogue_acc)})
    # acc = 0
    # dialogue_acc = 0
    # total = 0
    # for row in value:
    #     acc += row["Accuracy"]
    #     dialogue_acc += row["Dialogue Accuracy"]
    #     total += 1
    # final_table.append({"Data":key[0], "Model":key[1], "Accuracy":acc/total, "Dialogue Accuracy": dialogue_acc/total})





print(tabulate.tabulate(final_table, headers="keys", tablefmt="fancy_grid", floatfmt=".2f", showindex=False, stralign="center"))
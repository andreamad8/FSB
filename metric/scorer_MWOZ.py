from dataflow.multiwoz.execute_programs import exec_program_gen
from dataflow.multiwoz.create_belief_state_prediction_report import eval_state_report
from dataflow.multiwoz.evaluate_belief_state_predictions import evaluate
import jsons
import os
import glob
import jsonlines
import json
import collections
from tabulate import tabulate
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(files_test, files_to_score):
    with open(files_test, encoding="utf-8") as f:
        data_test = json.load(f)
    if type(files_to_score) == list:
        data_to_score = files_to_score
    else:
        with open(files_to_score, encoding="utf-8") as f:
            data_to_score = json.load(f)
        data_to_score = data_to_score["generation"]

    new_data_test = collections.defaultdict(dict)
    for d_test, d_to_score in zip(data_test,data_to_score):
        d_test["gen_query"] = d_to_score["query"]
        new_data_test[d_test["dialogue_id"]][d_test["turn_index"]] = d_test
    return new_data_test


def load_jsonl(files):
    with jsonlines.open(files) as reader:
        data = list(reader)
    return data

def plug_generation_to_data(gold_data, gen_data):
    # print(gold["dialogue_id"])
    for id_t, turns in enumerate(gold_data["turns"]):
        # print(turns)
        gold_data["turns"][id_t]['lispress'] = gen_data[gold_data["dialogue_id"]][id_t]["gen_query"]

    return gold_data

def random_file_name():
    import random
    import string
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

# truncate flot to 2 significant digits
def truncate(n, decimals=2):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

# for example 
# print(truncate(0.123456789, 2)) # 0.12 

def score_flowMWOZ(file_to_score):

    data = load_data('../data/flowMWOZ/test.json', file_to_score)

    # random file name
    temp_file = random_file_name()

    with open(temp_file, 'w') as f:
        for obj in load_jsonl("../data/flowMWOZ/test.dataflow_dialogues.jsonl"):
            new_obj = plug_generation_to_data(obj,data)
            f.write(json.dumps(new_obj) + '\n')


    # execute the programs
    # print("Executing the programs")
    complete_execution_results_file = exec_program_gen(dialogues_file=temp_file,
                                                        outbase=random_file_name())
    # print("Evaluating the results")
    belief_state_prediction_report_jsonl = eval_state_report(input_data_file=complete_execution_results_file,
                        file_format="dataflow",
                        remove_none= True,
                        gold_data_file="../data/flowMWOZ/test.belief_state_tracker_data.jsonl",
                        outbase=random_file_name())
    # print("Scoring the results")
    score = evaluate(prediction_report_jsonl=belief_state_prediction_report_jsonl, outbase="")

    # delete files
    os.remove(complete_execution_results_file)
    os.remove(belief_state_prediction_report_jsonl)
    os.remove(temp_file)

    return {"JGA":score.accuracy, **score.accuracy_for_slot}



results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
table = []
for file_name in tqdm(glob.glob("../generations/flowMWOZ*.json")):
    # read the file and check if JGA is in score dict
    if len(file_name.split("/")[-1].replace(".json","").split("_")) == 5:
        percentaege = 1.0
        _, shot, ranker, _, trial = file_name.split("/")[-1].replace(".json","").split("_")
    else:
        print(file_name)
        _, shot, _, ranker, percentaege, trial = file_name.split("/")[-1].replace(".json","").split("_")
    with open(file_name, encoding="utf-8") as f:
        data_score = json.load(f)
    with open(file_name, 'w') as f:
        json.dump(data_score, f, indent=4)
    if "JGA" not in data_score["score"]:
        JGA = score_flowMWOZ(file_name)["JGA"]
        data_score["score"]["JGA"] = JGA
        # save file with new score
        with open(file_name, 'w') as f:
            json.dump(data_score, f, indent=4)
    else:
        JGA = data_score["score"]["JGA"]
    results[ranker][shot][percentaege].append(JGA)
    table.append({"shot": shot, "trial": trial, "percentaege": percentaege, "JGA": JGA*100})
print(tabulate(table, headers="keys", tablefmt="fancy_grid", floatfmt=".2f", showindex=True, stralign="center"))



final_table = []
for ranker in results:
    for shot in results[ranker]:
        for percentaege in results[ranker][shot]:
            mean = np.mean(results[ranker][shot][percentaege])
            std = np.std(results[ranker][shot][percentaege])
            final_table.append({"ranker": ranker, "shot": shot, "percentaege": percentaege, "mean": mean*100, "std": std*100})

print(tabulate(final_table, headers="keys", tablefmt="fancy_grid", floatfmt=".2f", showindex=True, stralign="center"))




# plot results with seaborn and matplotlib
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

# prepare data for plotting
data = []
for ranker in results:
    for shot in results[ranker]:
        x = results[ranker][shot].keys()
        # convert x to float
        x = [float(i) for i in x]
        y = [np.mean(results[ranker][shot][percentaege])*100 for percentaege in results[ranker][shot]]
        yerr = [np.std(results[ranker][shot][percentaege])*100 for percentaege in results[ranker][shot]]

        data.append({"x": x, "y": y, "yerr": yerr, "ranker": ranker, "shot": shot})

# plot data with seaborn using line plot with fill_between
for d in data:
    # convert to numpy array
    x = np.array(d["x"])
    y = np.array(d["y"])
    yerr = np.array(d["yerr"])
    # plot
    ax.plot(x, y, label=d["ranker"]+"_"+d["shot"])
    ax.fill_between(x, y-yerr, y+yerr, alpha=0.2)

plt.legend(loc='lower right')
plt.xlabel("Percentage of training data")
plt.ylabel("JGA")
plt.title("JGA with error bars")
plt.savefig("JGA_error_bars.png")



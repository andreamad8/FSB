import json
import glob
import os
from official_coQA_scorer import CoQAEvaluator
import tabulate

# load CoQA test set and compute score
def loadCoQA(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

coQA_data = loadCoQA('../data/coQA/test.json')
evaluator = CoQAEvaluator('../data/coQA/coqa-dev-v1.0.json')

table = []
# load generated responses for CoQA
for file in glob.glob("../generations/coQA/coQA*.json"):
    with open(file, "r") as f:
        data = json.load(f)
        output_dic = []
        for pred_sample, gold_sample in zip(data["generation"], coQA_data):
            for answer, turn_id in zip(pred_sample["dialogue"],gold_sample["turn_id"]):
                output_dic.append({"id": gold_sample["id"], "turn_id": turn_id, "answer": answer[0]})

    if "cof1" not in data['score']:
        # save to json file
        with open("coQA_output.json", "w") as f:
            json.dump(output_dic, f, indent=4)



        pred_data = CoQAEvaluator.preds_to_dict("coQA_output.json")

        print(evaluator.model_performance(pred_data)["overall"])
        f1 = evaluator.model_performance(pred_data)["overall"]["f1"]
        em = evaluator.model_performance(pred_data)["overall"]["em"]
        table.append({"f1": f1, "em": em, "file": file})
        os.remove("coQA_output.json")

        data['score']['cof1'] = f1
        data['score']['coem'] = em
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        print("F1:", data['score']['cof1'])
        print("EM:", data['score']['coem'])

print(tabulate.tabulate(table, headers="keys", tablefmt="fancy_grid"))
# delete coQA_output.json
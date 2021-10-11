import glob
import json


files = ["tmp/train_9_2_roberta-large",
        "tmp/train_9_1_roberta-large",
        "tmp/train_9_0_roberta-large",
        "tmp/train_8_0_roberta-large",
        "tmp/train_8_1_roberta-large",
        "tmp/train_8_2_roberta-large",
        "tmp/train_7_2_roberta-large",
        "tmp/train_7_1_roberta-large",
        "tmp/train_7_0_roberta-large",
        "tmp/train_10_0_roberta-large",
        "tmp/train_10_1_roberta-large",
        "tmp/train_10_2_roberta-large",
        "tmp/train_10_2_roberta-base",
        "tmp/train_10_1_roberta-base",
        "tmp/train_10_0_roberta-base",
        "tmp/train_9_0_roberta-base",
        "tmp/train_9_1_roberta-base",
        "tmp/train_9_2_roberta-base",
        "tmp/train_8_2_roberta-base",
        "tmp/train_8_1_roberta-base",
        "tmp/train_8_0_roberta-base",
        "tmp/train_7_0_roberta-base",
        "tmp/train_7_1_roberta-base",
        "tmp/train_7_2_roberta-base"]

for fi in files: #glob.glob("tmp/*"):
    print(fi)
    _, shot, rep, model = fi.split("_")
    print(shot, rep, model)
    res = json.load(open(f"{fi}/eval_results.json","r"))
    temp = {
        "score": {
            "MacroF1": res["eval_MacroF1"]*100,
            "MicroF1": res["eval_MicroF1"]*100,
            "WeightedF1": res["eval_WeightedF1"]*100,
            "acc": res["eval_acc"]*100
        }
    }
    with open(f'../generations/skill-selector_{shot}_{model}_1_{rep}.json', 'w') as fp:
        json.dump(temp, fp, indent=4)



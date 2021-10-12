import json
from collections import defaultdict
from tabulate import tabulate
import numpy as np
import pandas as pd
import math
import glob

data = defaultdict(lambda: defaultdict(list))

for name in glob.glob('generations/msc-parse-dialogue*.json'):
    parse = name.replace("generation/","").replace(".json","").split("_")
    # print(parse)
    dataset, shot, model, beam_dosample, id_prefix = parse
    # print(dataset, shot, model, id_prefix)
    
    data[shot][f"{model}_{id_prefix}"].append(name)


def save_file(filename, results):
    filename = filename.replace("EleutherAI/","").replace("../few-shot-lm/","")
    with open(f'generations/{filename}', 'w') as fp:
        json.dump(results, fp, indent=4)

for shot, v in data.items():
    
    if int(shot) in [0,1,3]:
        for model_name, list_file in v.items():
            model, id_prefix = model_name.split("_")
            print(model_name)
            print(list_file)
            dict_res = {
                "score": {
                    "B4": 0.0,
                    "F1": 0.0,
                    "RL": 0.0,
                    "ppl": 0.0
                },
                "generation": []
            }
            generation = []
            B4 = []
            F1 = []
            RL = []
            ppl = []
            for file_name in list_file:
                res = json.load(open(file_name,"r"))
                ppl.append(res['score']["ppl"])
                B4.append(res['score']["B4"])
                F1.append(res['score']["F1"])
                RL.append(res['score']["RL"])
                generation += res["generation"]
            dict_res['generation'] = generation
            dict_res['score']['ppl'] = np.mean(ppl)
            dict_res['score']['B4'] = np.mean(B4)
            dict_res['score']['F1'] = np.mean(F1)
            dict_res['score']['RL'] = np.mean(RL)
            save_file(f"msc-parse_{shot}_{model}_False_{id_prefix}.json", dict_res)
            # input()   
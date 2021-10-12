import glob
import json
from collections import defaultdict
from tabulate import tabulate
import numpy as np
import pandas as pd
import math
from metric.smd_scorer import score_SMD


data = defaultdict(lambda: defaultdict(list))

for name in glob.glob('generations/smd-*.json'):
    parse = name.replace("generation/","").replace(".json","").split("_")
    # print(parse)
    dataset, shot, model, beam_dosample, id_prefix = parse
    # print(dataset, shot, model, id_prefix)

    data[f"{model}_{id_prefix}"][shot].append(name)

def save_file(filename, results):
    filename = filename.replace("EleutherAI/","").replace("../few-shot-lm/","")
    with open(f'generations/{filename}', 'w') as fp:
        json.dump(results, fp, indent=4)

for k, v in data.items():
    model, id_prefix = k.split("_")
    for shots, list_file in v.items():
        dict_res = {
            "score": {
                "BLEU": 0.0,
                "F1": 0.0,
                "F1 navigate": 0.0,
                "F1 weather": 0.0,
                "F1 schedule": 0.0,
            },
            "generation": []
        }

        # follow this sequence cause test test is nasty
        # schedule
        # navigate
        # weather
        temp_gen = {}
        ppl = []
        for file_name in list_file:
            res = json.load(open(file_name,"r"))
            print(file_name)
            print(res['score'])
            ppl.append(res['score']["ppl"])
            if not np.isnan(res['score']['F1 navigate']):
                dict_res['score']["F1 navigate"] = res['score']["F1 navigate"]
                temp_gen["navigate"] = res["generation"]
            if not np.isnan(res['score']['F1 weather']):
                dict_res['score']["F1 weather"] = res['score']["F1 weather"]
                temp_gen["weather"] = res["generation"]
            if not np.isnan(res['score']['F1 schedule']):
                dict_res['score']["F1 schedule"] = res['score']["F1 schedule"]
                temp_gen["schedule"] = res["generation"]
        dict_res["generation"] = temp_gen["schedule"]+temp_gen["navigate"]+temp_gen["weather"]
        res = score_SMD(dict_res["generation"], "data/smd/test.json")
        dict_res['score'] = res
        dict_res['score']['ppl'] = np.mean(ppl)
        save_file(f"smd_{shots}_{model}_{beam_dosample}_{id_prefix}.json", dict_res)
        # input()
        # print(list_file)
    print()
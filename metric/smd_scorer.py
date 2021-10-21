import json
import os.path
from metric.bleu import moses_multi_bleu
import glob as glob
import numpy as np
import jsonlines
from tabulate import tabulate
from tqdm import tqdm

def compute_prf_SMD(gold, pred, global_entity_list):#, kb_plain=None):
    # local_kb_word = [k[0] for k in kb_plain]
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in set(pred):
            if p in global_entity_list:# or p in local_kb_word:
                if p not in gold:
                    FP += 1
                    print(p)
        # print("TP",TP)
        # print("FP",FP)
        # print("FN",FN)
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        # print(precision)
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        # print(recall)
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        # print(F1)
    else:
        precision, recall, F1, count = 0, 0, 0, 0
    return F1, count

def get_global_entity_KVR():
    with open('data/smd/kvret_entities.json') as f:
        global_entity = json.load(f)
        global_entity_list = []
        for key in global_entity.keys():
            if key != 'poi':
                global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
            else:
                for item in global_entity['poi']:
                    global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
        global_entity_list = list(set(global_entity_list))
    return global_entity_list

def post_process_GPT(text):
    return text.replace("."," .").replace("'"," '").replace("?"," ?").replace(","," ,").replace("!"," !").replace("  "," ")
    


def score_SMD(file_to_score, file_to_gold):
    if type(file_to_score) == list:
            genr_json = file_to_score
    else:
        with open(file_to_score, encoding="utf-8") as f:
            genr_json = json.load(f)
        genr_json = genr_json['generation']

    gold_json = json.load(open(file_to_gold,"r"))
    global_entity_list = get_global_entity_KVR()
    GOLD, GENR = [], []
    F1_score = []
    F1_domain = {"navigate":[],"weather":[],"schedule":[]}

    for gold_dial, pred_diag in zip(gold_json, genr_json):
        for id_turn, (turn_gold, turn_pred) in enumerate(zip(gold_dial["dialogue"], pred_diag["dialogue"])):
            # print(id_turn,turn_gold, turn_pred)
            F1, count = compute_prf_SMD(gold_dial["gold_ent"][id_turn], post_process_GPT(turn_pred[0]), global_entity_list)
            if(count==1): 
                F1_score.append(F1)
                F1_domain[gold_dial["domain"]].append(F1)
            GOLD.append(turn_gold[1])
            GENR.append(post_process_GPT(turn_pred[0]))

    BLEU = moses_multi_bleu(np.array(GENR),np.array(GOLD))

    return {"BLEU":BLEU, 
            "F1":100*np.mean(F1_score), 
            "F1 navigate":100*np.mean(F1_domain["navigate"]), 
            "F1 weather":100*np.mean(F1_domain["weather"]), 
            "F1 schedule":100*np.mean(F1_domain["schedule"])}

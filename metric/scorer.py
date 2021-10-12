
import json
import numpy as np
from metric.bleu import moses_multi_bleu
from metric.smd_scorer import score_SMD
from metric.general import Rouge_L, BLEU_4, get_F1, feqa_scorer

def load_data(files_test, files_to_score, meta_type):
    with open(files_test, encoding="utf-8") as f:
        data_test = json.load(f)
    if type(files_to_score) == list:
        data_to_score = files_to_score
    else:
        with open(files_to_score, encoding="utf-8") as f:
            data_to_score = json.load(f)
        data_to_score = data_to_score["generation"]

    GOLD, GENR = [], []
    if meta_type == "last_turn":
        dict_data_to_score = {d['meta']: d for d in data_to_score}
        for d in data_test:
            ## take the 1th utterance of the 2th turn
            if d['meta'] in dict_data_to_score:
                GOLD.append(d['dialogue'][1][0])
                GENR.append(dict_data_to_score[d['meta']]['dialogue'][1][0])
    elif meta_type == "KB":
        for d_test, d_to_score in zip(data_test,data_to_score):
            for turn_test, turn_to_score in zip(d_test["KB"], d_to_score["dialogue"]):
                GOLD.append("None" if len(turn_test) == 0 else turn_test[0])
                GENR.append("None" if len(turn_to_score) == 0 else turn_to_score[0])
    else:
        for d_test, d_to_score in zip(data_test,data_to_score):
            for turn_test, turn_to_score in zip(d_test["dialogue"], d_to_score["dialogue"]):
                if meta_type == "all_turns":
                    GOLD.append(turn_test[0])
                    GENR.append(turn_to_score[0])
                    if turn_test[1]!="":
                        GOLD.append(turn_test[1])
                        GENR.append(turn_to_score[1])
                else:
                    GOLD.append(turn_test[1])
                    GENR.append(turn_to_score[0])
    return GOLD, GENR

def score(files_test, files_to_score, meta_type):
    GOLD, GENR = load_data(files_test,files_to_score, meta_type)
    print("Evaluating ROUGE-L")
    RL = Rouge_L(GOLD, GENR)
    print("Evaluating B4")
    B4 = BLEU_4(GOLD, GENR)
    print("Evaluating BLUE avg")
    BLEU = moses_multi_bleu(np.array(GENR),np.array(GOLD))
    print("Evaluating F1")
    f1 = get_F1(GENR,GOLD)
    if "smd" in files_test:
        res = score_SMD(files_to_score, files_test)
        return res
    if "wow" in files_test or "wit" in files_test:
        GOLD, GENR = load_data(files_test,files_to_score, meta_type="KB")
        kf1 = get_F1(GENR,GOLD)
        return {"BLEU":BLEU, "B4":B4*100,"F1":f1*100, "RL":RL*100,"kf1":kf1*100}
    if "dialKG" in files_test:
        feqa_res =, 0.0
        # feqa_res, = feqa_scorer(files_test,files_to_score)
        return {"BLEU":BLEU, "B4":B4*100,"F1":f1*100, "RL":RL*100,"feqa":feqa_res}

    return {"BLEU":BLEU, "B4":B4*100,"F1":f1*100, "RL":RL*100}


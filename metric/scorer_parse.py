import json
import numpy as np
from collections import Counter
import glob
from tqdm import tqdm
import re
from dictdiffer import diff
from metric.calculator import evaluate_predictions

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s

def Rouge_L(GOLD, GENR):
    rouge_score = []
    for guess, answ in zip(GOLD, GENR):
        rouge_score.append(rouge_PARAI(guess=guess, answers=[answ])['rouge-l'])
    return np.mean(rouge_score)


def rouge_PARAI(guess: str, answers: str):
        """
        Compute ROUGE score between guess and *any* answer.
        Done with compute_many due to increased efficiency.
        :return: (rouge-1, rouge-2, rouge-L)
        """
        # possible global initialization
        try:
            import rouge
        except ImportError:
            # User doesn't have py-rouge installed, so we can't use it.
            # We'll just turn off rouge computations
            return None, None, None


        evaluator = rouge.Rouge(
            metrics=['rouge-n', 'rouge-l'], max_n=2
        )
        scores = [
            evaluator.get_scores(
                normalize_answer(guess), normalize_answer(a)
            )
            for a in answers
        ]
        scores_rouge1 = max(score['rouge-1']['r'] for score in scores)
        scores_rouge2 = max(score['rouge-2']['r'] for score in scores)
        scores_rougeL = max(score['rouge-l']['r'] for score in scores)
        return {
                    'rouge-1':scores_rouge1,
                    'rouge-2':scores_rouge2,
                    'rouge-l':scores_rougeL,
                }

def BLEU_4(GOLD, GENR):
    BLEUscore = []
    for guess, answ in zip(GOLD, GENR):
        BLEUscore.append(computeBLUEPARLAI(guess=guess, answers=[answ]))
    return np.mean(BLEUscore)


def computeBLUEPARLAI(guess: str, answers: str, k: int = 4):
    try:
        from nltk.translate import bleu_score as nltkbleu
    except ImportError:
        # User doesn't have nltk installed, so we can't use it for bleu
        # We'll just turn off things, but we might want to warn the user
        return None

    weights = [1 / k for _ in range(k)]
    score = nltkbleu.sentence_bleu(
        [normalize_answer(a).split(" ") for a in answers],
        normalize_answer(guess).split(" "),
        smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
        weights=weights,
    )
    return score   

def load_data(files_test, files_to_score, key="meta"):
    with open(files_test, encoding="utf-8") as f:
        data_test = json.load(f)
    if type(files_to_score) == list:
        data_to_score = files_to_score
    else:
        with open(files_to_score, encoding="utf-8") as f:
            data_to_score = json.load(f)
        data_to_score = data_to_score["generation"]

    GOLD, GENR = [], []
    if key == "last_turn":

        for d_test, d_score in zip(data_test, data_to_score):
            gold_query = d_test["dialogue"][-1][1]
            GOLD.append(gold_query)
            GENR.append(d_score["meta"])
            # break
    elif key == "meta":
        for d_test, d_score in zip(data_test, data_to_score):
            GOLD.append(d_test["meta"])
            GENR.append(d_score["meta"])
    elif key == "sentence":
        for d_test, d_score in zip(data_test, data_to_score):
            GOLD.append(d_test["query"])
            GENR.append(d_score["query"])
    elif key == "dialKG":
        for d_test, d_score in zip(data_test, data_to_score):
            if len(d_test["query"]) == 1:
                query = "\t".join(d_test["query"][0])
            else:
                query = "\t".join(d_test["query"][0]) + "\t\t" + "\t".join(d_test["query"][1]) 
            GOLD.append(query)
            GENR.append(d_score["query"])
    else:
        # assert len(data_test) == len(data_to_score)
        for d_test, d_to_score in zip(data_test,data_to_score):
            for meta_test, meta_to_score in zip(d_test[key], d_to_score[key]):
                GOLD.append("none" if len(meta_test)==0 else meta_test[0])
                GENR.append(meta_to_score[0])
    return GOLD, GENR


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def get_F1(pred,gold):
    f1 = []
    for p,g in zip(pred,gold):
        f1.append(_prec_recall_f1_score(normalize_answer(p).split(),normalize_answer(g).split()))
    return np.mean(f1)


def compute_JGA(files_test, files_to_score):
    with open(files_test, encoding="utf-8") as f:
        data_test = json.load(f)
    if type(files_to_score) == list:
        data_to_score = files_to_score
    else:
        with open(files_to_score, encoding="utf-8") as f:
            data_to_score = json.load(f)

    JGA = []
    SLOT_ACC = []
    for d_test, d_to_score in zip(data_test,data_to_score):
        assert len(d_test["state"]) == len(d_to_score["state"])
        for state_test, state_to_score in zip(d_test["state"], d_to_score["state"]):
            if state_test != "none" and len(state_test)>0:
                GOLD_STATE = {sv.split("=")[0].replace("_"," ") : sv.split("=")[1] for sv in state_test.split("\t")}
                PRED_STATE = state_to_score
                diff_state = list(diff(GOLD_STATE,PRED_STATE))
                if len(diff_state) == 0:
                    JGA.append(1)
                else:
                    JGA.append(0)

                for slot, value in GOLD_STATE.items():
                    if slot in PRED_STATE:
                        if PRED_STATE[slot] == value:
                            SLOT_ACC.append(1)
                        else:
                            SLOT_ACC.append(0)
                    else:
                        SLOT_ACC.append(0)

                # print(GOLD_STATE)
                # print(PRED_STATE)
                # print(JGA[-1])
                # print(SLOT_ACC[-1])
    return np.mean(JGA), np.mean(SLOT_ACC)

# https://stackoverflow.com/questions/28734607/evaluation-of-lists-avgpk-and-rk-are-they-same
def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result

def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result

def compute_recall_k(GENR,GOLD):
    recallk_path = {1:[],3:[],5:[],10:[],25:[]}
    recallk_ent = {1:[],3:[],5:[],10:[],25:[]}
    for (path, entities), gold_path in zip(GENR,GOLD):

        for k in [1,3,5,10,25]:
            if gold_path == "none" and path.lower() != "none":
                recallk_path[k].append(0)
                recallk_ent[k].append(0)    
            elif gold_path != "none" and path.lower() == "none":
                recallk_path[k].append(0)
                recallk_ent[k].append(0)    
            else:
                predicted = [e[0].lower() for e in entities]
                actual = [gold_path.split("\t")[-1].lower()]
                recallk_ent[k].append(recall(actual, predicted, k))

                path_without_last = path[:path.rfind("\t")+1]
                predicted = [path_without_last.lower().replace("~","")+e[0].lower() for e in entities]
                actual = [gold_path.lower().replace("~","")]
                recallk_path[k].append(recall(actual, predicted, k))

    return {f"path_{k}":np.mean(v)*100 for k,v in recallk_path.items()}, {f"ent_{k}":np.mean(v)*100 for k,v in recallk_ent.items()}

def compute_acc(pred,gold):
    acc = []
    for p,g in zip(pred,gold):
        if normalize_answer(p) == normalize_answer(g):
            acc.append(1)
        else:
            acc.append(0)
    return np.mean(acc)

def score(files_test, files_to_score, meta_type):
    if "dialKG" in meta_type:
        GOLD, GENR = load_data(files_test,files_to_score, key="dialKG")
        recallk_path, recallk_ent = compute_recall_k(GENR,GOLD) 
        return {**recallk_path,**recallk_ent}
    elif meta_type in ["top", "flowMWOZ", "semflow"]:
        GOLD, GENR = load_data(files_test,files_to_score, key="sentence")
    else:
        GOLD, GENR = load_data(files_test,files_to_score, key="meta")

    print("Evaluating ROUGE-L")
    RL = Rouge_L(GOLD, GENR)
    print("Evaluating B4")
    B4 = BLEU_4(GOLD, GENR)
    print("Evaluating F1")
    f1 = get_F1(GENR,GOLD)

    if meta_type == "top":
        acc = evaluate_predictions(GOLD, GENR)
        return {"B4":B4*100,"F1":f1*100, "RL":RL*100,**acc} 

    if meta_type in ["flowMWOZ", "semflow"]:
        acc = 0.0
        for g, gt in zip(GENR,GOLD):
            if g.replace(" ","") == gt.replace(" ",""):
                acc += 1
        acc = acc/len(GENR)
        return {"B4":B4*100,"F1":f1*100, "RL":RL*100,"acc":acc} 


    if "wit" in meta_type or "wow" in meta_type:
        acc = compute_acc(GENR,GOLD)

        return {"B4":B4*100,"F1":f1*100, "RL":RL*100,"acc":acc*100}

    if "mwoz" in meta_type:
        JGA, SLOT_ACC = compute_JGA(files_test,files_to_score)
        return {"B4":B4*100,"F1":f1*100, "RL":RL*100, "JGA":JGA*100,"SLOT_ACC":SLOT_ACC*100}
    return {"B4":B4*100,"F1":f1*100, "RL":RL*100}


if __name__ == "__main__":
    table = []
    for file in tqdm(glob.glob("generations/dialKG-parse_*.json")):
        # print(file)
        sco = score("data/dialKG/test.json",file,"dialKG")
        with open(file, encoding="utf-8") as f:
            data_test = json.load(f)
        ppl = data_test["score"]['ppl']
        sco["ppl"] = ppl
        data_test["score"] = sco
        with open(file, 'w') as fp:
            json.dump(data_test, fp, indent=4)

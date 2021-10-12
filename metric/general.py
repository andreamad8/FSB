from collections import Counter
import glob
from collections import defaultdict
from tabulate import tabulate
import nltk
import json
import benepar
from tqdm import tqdm
import re
from nltk import word_tokenize
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### this function are taken from ParlAI
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

def convert_triples_to_sent(triples):
    if len(triples) == 0: return ""
    triples = [tri.split("\t") for tri in triples[0].split("\t\t")]
    sentences_triple = []
    for tri in triples:
        rel = tri[1]

        if rel.startswith("~"):
            rel = rel[1:]
            subject = tri[2]
            object = tri[0]
        else:
            subject, object = tri[0], tri[2]
        sentence_triple = " ".join([subject, rel, object])
        sentences_triple.append(sentence_triple)
    sentences_triple = ". ".join(sentences_triple)

    return sentences_triple


## To evaluate FEQA uncomment this
# scorer = FEQA(use_gpu=True)
def feqa_scorer(files_test, files_to_score):
    with open(files_test, encoding="utf-8") as f:
        data_test = json.load(f)
    if type(files_to_score) == list:
        data_to_score = files_to_score
    else:
        with open(files_to_score, encoding="utf-8") as f:
            data_to_score = json.load(f)
        data_to_score = data_to_score["generation"]

    documents = []
    summaries = []
    for d_test, d_to_score in zip(data_test,data_to_score):
        for turn_KB, turn_test, turn_to_score in zip(d_test["KG"], d_test["dialogue"], d_to_score["dialogue"]):
            documents.append(turn_test[0]+" "+convert_triples_to_sent(turn_KB))
            summaries.append(turn_to_score[0])

    assert len(documents) == len(summaries)
    return scorer.compute_score(documents, summaries, aggregate=True)

def argmin(a):
    return min(range(len(a)), key=lambda x: a[x])

def metric_report(y_test, y_pred,verbose=False):
    #importing confusion matrix
    # confusion = confusion_matrix(y_test, y_pred)
    # print('Confusion Matrix\n')
    # print(confusion)

    #importing accuracy_score, precision_score, recall_score, f1_score
    if verbose:
        print('Accuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

        print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
        print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
        print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

        print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
        print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
        print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

        print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
        print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
        print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

    # from sklearn.metrics import classification_report
    # print('\nClassification Report\n')
    # print(classification_report(y_test, y_pred))
    return {"acc":accuracy_score(y_test, y_pred),
            "MicroF1":f1_score(y_test, y_pred, average='micro'),
            "MacroF1":f1_score(y_test, y_pred, average='macro'),
            "WeightedF1":f1_score(y_test, y_pred, average='weighted')
            }
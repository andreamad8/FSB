import jsonlines
import json
from nltk.tokenize import sent_tokenize

def preproc(split):
    data = []
    with jsonlines.open(f'{split}.json') as reader:
        for obj in reader:
            dial = {"meta":[],"dialogue":[]}
            for ids_t, turn in enumerate(obj["dialogue"]):
                if ids_t % 2 == 0:
                    temp = [turn["text"]]
                    temp_meta = [{"emotion":turn["emotion"], "act": turn["act"]}]
                else:
                    temp.append(turn["text"])
                    temp_meta.append({"emotion":turn["emotion"], "act": turn["act"]})
                    dial["dialogue"].append(temp)
                    dial["meta"].append(temp_meta)

            if len(temp) == 1:
                temp.append("")
                dial["dialogue"].append(temp)
                dial["meta"].append({})

            assert len(dial["meta"]) == len(dial["dialogue"])
            data.append(dial)

    split = split.replace("_original","")
    with open(f'{split}.json', 'w') as fp:
        json.dump(data, fp, indent=4)

preproc("valid_original")
preproc("test_original")
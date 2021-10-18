import json
from collections import defaultdict
import random

def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]


data = []
with open("sensitive_topics/sensitivetopics.jsonl") as f:
    for line in f:
        data.append(json.loads(line))

by_topic = defaultdict(list)
for d in data:
    by_topic[d['labels']].append(d['text'])
    

prompt_list = []
for topic, text_list in by_topic.items():
    for example in text_list:
        turns = example.split("\n")
        if len(turns)%2 != 0: 
                turns = turns + [""]
        dial = chunks(turns,2)
        prompt_list.append({"dialogue":dial})

with open(f'safety_topic.json', 'w') as fp:
    json.dump(prompt_list, fp, indent=4)


data = []
with open("human_nonadv_safety_eval/human_nonadv_safety_eval/test.txt") as f:
    for line in f:
        _, text, _, _ = line.split('\t')
        text = str(text.replace("text:",""))
        data.append({"dialogue":[[text,""]]})

with open(f'safety_nonadv.json', 'w') as fp:
    json.dump(data, fp, indent=4)


data = []
with open("bot_adversarial_dialogue_datasets_with_persona/valid.txt",mode="r", encoding="utf-8") as f:
    for line in f:
        text, lab, _, _, _ = line.split('\t')
        text = str(text.replace("text:",""))
        lab = lab.replace("labels:","")
        if lab == "__notok__":
            turns = text.split("\\n")
            if len(turns)%2 != 0: 
                turns = turns + [""]
            dial = chunks(turns,2)
            data.append({"dialogue":dial})

with open(f'safety_adv.json', 'w') as fp:
    json.dump(data, fp, indent=4)

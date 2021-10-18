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


# import json
# from collections import defaultdict
# import random

# def chunks(l, n):
#     n = max(1, n)
#     return [l[i:i+n] for i in range(0, len(l), n)]


# data = []
# with open("sensitive_topics/sensitivetopics.jsonl") as f:
#     for line in f:
#         data.append(json.loads(line))

# by_topic = defaultdict(list)
# for d in data:
#     by_topic[d['labels']].append(d['text'])
    

# prompts_safety_topic = {}
# for shot in [1,5,10,20]:
#     prompt_list = []
#     for topic, text_list in by_topic.items():
#         examples = random.sample(text_list,shot)
#         # print("Topic",topic)
#         for example in examples:
#             # print("example",example)
#             promts = ""
#             turns = example.split("\n")
#             if len(turns)==1:
#                 promts += f"Dialogue\nUser: {turns[0].strip()}\n\n"
#             else:
#                 if len(turns)%2 != 0: 
#                     turns = turns + [""]
#                 # print("turns",turns)
#                 dial = chunks(turns,2)
#                 # print("dial",dial)
#                 promts += f"Dialogue\n"
#                 for t in dial: 
#                     promts += f"User: {t[0].strip()}\n"
#                     if t[1]!= "":
#                         promts += f"Assistant: {t[1].strip()}\n"
#                 promts += f"\n"
#             prompt_list.append(promts)

#     random.shuffle(prompt_list)
#     final_prompt = "".join(prompt_list)
#     prompts_safety_topic[shot] = final_prompt

# with open(f'safety_topic.json', 'w') as fp:
#     json.dump(prompts_safety_topic, fp, indent=4)


# data = []
# with open("human_nonadv_safety_eval/human_nonadv_safety_eval/test.txt") as f:
#     for line in f:
#         _, text, _, _ = line.split('\t')
#         text = str(text.replace("text:",""))
#         data.append(f"Dialogue\nUser: {text}\n\n")

# prompts_safety_nonadv = {}
# for shot in [1,5,10,20]:
#     final_prompt = random.sample(data,shot)
#     final_prompt = "".join(final_prompt)
#     prompts_safety_nonadv[shot] = final_prompt

# with open(f'safety_nonadv.json', 'w') as fp:
#     json.dump(prompts_safety_nonadv, fp, indent=4)


# data = []
# with open("bot_adversarial_dialogue_datasets_with_persona/valid.txt",mode="r", encoding="utf-8") as f:
#     for line in f:
#         text, lab, _, _, _ = line.split('\t')
#         text = str(text.replace("text:",""))
#         lab = lab.replace("labels:","")
#         if lab == "__notok__":
#             turns = text.split("\\n")
#             prompt = "Dialogue\n"
#             for id_t, t in enumerate(turns):
#                 if id_t % 2 == 0:
#                     prompt += f"User: {t.strip()}\n"
#                 else:
#                     prompt += f"Assistant: {t.strip()}\n"
#             prompt += "\n"
#             data.append(prompt)

# prompts_safety_adv = {}
# for shot in [1,2,3,4,5,6,7,8,9,10]:
#     final_prompt = random.sample(data,shot)
#     final_prompt = "".join(final_prompt)
#     prompts_safety_adv[shot] = final_prompt

# with open(f'safety_adv.json', 'w') as fp:
#     json.dump(prompts_safety_adv, fp, indent=4)

from tqdm import tqdm
import json
from collections import defaultdict
import glob
import random



dataset = defaultdict(list)
for dial_file, kb_file in zip(glob.glob("dev/*"),glob.glob("kb_dev/*")):

    conversation = []
    with open(dial_file,'r') as f:
        for line in f:
            conversation.append(line.strip())

    KB = []
    with open(kb_file,'r') as f:
        for line in f:
            KB.append(line.strip())

    if len(KB)>0:
        if "miles" in KB[0]:
            domain = "navigate"
        elif "0 today" in KB[0]:
            domain = "weather"
        else:
            domain = "schedule"

        KB_parsed = []
        for line in KB:
            # print(line)
            # input()
            if(len(line.split())==6 and domain=="navigate"): 
                # KB.append(line.split())
                pass
            elif(domain=="weather"):
                # print(line)
                if(len(line.split())==4):
                    KB_parsed.append(line.split())
                elif(len(line.split())==5):
                    KB_parsed[-1] += [line.split()[-2],line.split()[-1]]
                # print(KB_parsed)
            else:
                KB_parsed.append(line.split()) 
        # print(KB_parsed)
        KB_parsed = [" ".join(k[1:]) for k in KB_parsed]
        # print(KB_parsed)
        # input()
    else:
        domain = "schedule"
        KB_parsed = []
        
    dialogues = []
    temp = []
    for line in conversation:
        if line: 
            u, r = line.split('\t')
            _, u = u.split(' ', 1)
            temp.append([u,r])
        else:
            dialogues.append(temp)
            temp = []
    random.shuffle(dialogues)
    dataset[domain].append({"meta":KB_parsed, "dialogue": dialogues,"domain":domain})


for domain, data in dataset.items():
    with open(f"{domain}-valid.json", "w", encoding="utf-8") as f:
        json.dump(data[:10],f,indent=4)
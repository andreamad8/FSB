from tqdm import tqdm
import json
from collections import defaultdict

def generate_dataset(data_split):
    num_lines = sum(1 for line in open(data_split,'r'))
    with open(data_split,'r') as f:
        conversation = []
        data = []
        KB = []
        idd = 0
        for line in tqdm(f,total=num_lines):
            line = line.strip()
            if line:
                if '#' in line:
                    if(idd!=0):
                        # dialogue = get_dialogue(conversation,tokenizer)
                        KB = [" ".join(k) for k in KB]
                        data.append({'id':idd,"domain":task_type,"dialogue":conversation, "KB":KB})
                    idd += 1
                    conversation = []
                    KB = []
                    line = line.replace("#","")
                    task_type = line
                    continue

                _, line = line.split(' ', 1)
                if '\t' in line:
                    # print(line)
                    u, r, gold_ent = line.split('\t')
                    conversation.append({"spk":"USR","text":u})
                    conversation.append({"spk":"SYS","text":r, "gold_ent":gold_ent})
                else:
                    if(len(line.split())==5 and task_type=="navigate"): 
                        # KB.append(line.split())
                        pass
                    elif(task_type=="weather"):
                        if(len(line.split())==3):
                            KB.append(line.split())
                        elif(len(line.split())==4):
                            KB[-1] += [line.split()[-2],line.split()[-1]]
                    else:
                        KB.append(line.split()) 
    dataset = defaultdict(list)
    cnt = 0
    for d in data:
        dialogue = {"meta":[], "dialogue": [],"domain":"", "gold_ent":[]}
        for i in range(0,len(d['dialogue']),2):
            # if(i+1>=len(d)): break
            if (d['dialogue'][i]["spk"] != 'USR'): 
                cnt += 1
                break
            if (d['dialogue'][i+1]["spk"] != 'SYS'): 
                cnt += 1
                break
            dialogue["dialogue"].append([d['dialogue'][i]['text'],d['dialogue'][i+1]['text']])
            # print(eval(d['dialogue'][i+1]["gold_ent"]))
            dialogue["gold_ent"].append(eval(d['dialogue'][i+1]["gold_ent"]))
        dialogue['meta'] = d['KB']
        dialogue['domain'] = d["domain"]
        dataset[d["domain"]].append(dialogue)
        # print(dialogue)
        # dataset.append(dialogue)
    return dataset 

# train = generate_dataset("train_o.txt")
# valid = generate_dataset("valid_o.txt")
test_by_domain = generate_dataset("dev.txt")
# generated = generate_dataset("weather_dialogues_alibaba.txt")

# with open("train.json", "w", encoding="utf-8") as f:
#     json.dump(train,f,indent=4)
# with open("valid.json", "w", encoding="utf-8") as f:
#     json.dump(valid,f,indent=4)

data_tot = []
for domain, data in test_by_domain.items():
    data_tot += data
    # if domain == "schedule":
    print(domain)
    with open(f"{domain}-train.json", "w", encoding="utf-8") as f:
        json.dump(data,f,indent=4)

# with open("train.json", "w", encoding="utf-8") as f:
#     json.dump(data_tot,f,indent=4)
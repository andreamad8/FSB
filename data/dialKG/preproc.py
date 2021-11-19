import csv
import json
from tqdm import tqdm
import random

def preprocess_dialogue(conversation):
    dials = []
    idx = 0
    for i,c in enumerate(conversation):
        dials.append({"speaker":c['speaker'],"gold_KB":c["gold-kb"], "text":c["text"].replace("  "," ")})
    return dials

# return list of chunks
def chunks(l, n):
    data = []
    for i in range(0, len(l), n):
        data.append(l[i:i + n])
    return data


def generate_dataset(file_name):
    data = []
    num_lines = sum(1 for line in open(file_name,'r'))
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        line_count = 0
        for row in tqdm(csv_reader,total=num_lines):
            conversation = []
            KB = []
            for t in eval(row[0]):
                if('message' in t):
                    conversation.append({"speaker":t['sender'], "text":t['message'], "gold-kb":[]})
                else:
                    if "path" in t['metadata']:
                        if(len(conversation)>0):
                            conversation[-1]["gold-kb"] = t['metadata']['path'][1]
                        
                        KB += t['metadata']['path'][1]
            dialogue = preprocess_dialogue(conversation)

            if len(dialogue)%2==1:
                dialogue.append({"speaker":"dummy", "text":"", "gold_KB":[]})
            # chunk the dialogue
            dialogue = chunks(dialogue,2)

            # convert the dialogue in the rigth format
            temp = {"dialogue": [], "query": []}
            # assign a random alphanumeric id to the dialogue
            temp["id"] = str(random.randint(0,1000000)) 
            for d in dialogue:
                temp["dialogue"].append([d[0]["text"],d[1]["text"]])
                if d[0]["gold_KB"]!=[] and d[1]["gold_KB"]!=[]:
                    temp["query"].append([d[0]["gold_KB"],d[1]["gold_KB"]])
                elif d[0]["gold_KB"]!=[] and d[1]["gold_KB"]==[]:
                    temp["query"].append([d[0]["gold_KB"],""])
                elif d[0]["gold_KB"]==[] and d[1]["gold_KB"]!=[]:
                    temp["query"].append(["",d[1]["gold_KB"]])
                else:
                    temp["query"].append(["",""])

            data.append(temp)



            line_count += 1
            # if(line_count == 10):break
        print(f'Processed {line_count} lines.')
    return data

split = ["train", "valid", "test"]
for s in split:
    data = generate_dataset(f"data/{s}.csv")
    # save the data in json format
    with open(f"{s}.json", 'w') as outfile:
        json.dump(data, outfile, indent=4)
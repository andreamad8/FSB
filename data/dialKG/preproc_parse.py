import json
import copy
def convert_dialogue_to_parsing_format(dialogue):
    """
    Convert the dialogue in the right format
    """
    data = []
    for dial in dialogue:
        # convert the dialogue in the rigth format
        dial_temp = []
        for turn, query in zip(dial["dialogue"],dial["query"]):
            dial_temp.append([turn[0]])
            if query[0]!="":
                data.append({"dialogue": copy.deepcopy(dial_temp), "query": query[0], "id": dial["id"]})
            if turn[1]!="":
                dial_temp[-1].append(turn[1])
            if query[1]!="":
                data.append({"dialogue": copy.deepcopy(dial_temp), "query": query[1], "id": dial["id"]})

    return data

splits = ["train","valid","test"]
for split in splits:
    with open(split+".json","r") as f:
        dialogue = json.load(f)
    data = convert_dialogue_to_parsing_format(dialogue)
    with open("parse-"+split+".json","w") as f:
        json.dump(data,f, indent=4)


        
    

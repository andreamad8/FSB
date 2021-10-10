import json
import copy
import random

data = json.load(open("test.json","r"))

samples = []
for dial in data:
    meta = dial["meta"]
    temp = []
    for id_turn, (turns, query) in enumerate(zip(dial['dialogue'], dial["query"])):
        temp.append([turns[0],""])
        if len(query)>0:
            temp[-1][1] = query[0]
            samples.append({"id_turn":id_turn,"id":dial["id"],"meta_info":dial["meta"], "meta":[[query[0]]],"dialogue":copy.deepcopy(temp[-3:])})
            temp[-1][1] = turns[1]
        else:
            temp[-1][1] = turns[1]
        
# random.shuffle(samples)
with open(f'parse-valid.json', 'w') as fp:
    json.dump(samples, fp, indent=4)

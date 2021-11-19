import json
from tqdm import tqdm
import jsonlines
import json
from nltk.tokenize import sent_tokenize
# from KILT.kilt.knowledge_source import KnowledgeSource
from collections import defaultdict
import random


def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

def preproc(split):
    data = []
    # ks = KnowledgeSource()
    # dial = {"id":obj['id'],"meta":[],"dialogue":[]}
    dialogue_by_id = defaultdict(lambda: defaultdict(list))
    with jsonlines.open(f'{split}') as reader:
        for obj in reader:
            # id_dialogue, turn = obj['id'].split("_")
            # print(obj["input"].split("\n"))

            if 'provenance' in obj["output"][0]:
                query = obj["output"][0]['provenance'][0]['title']
                # print(query)
                # print(chunks(obj["input"].split("\n")+[query],2))
                # # print(query)
                # page = ks.get_page_by_id(obj["output"][0]['provenance'][0]['wikipedia_id'])
                # # page = ks.get_page_by_title(query)
                # assert obj["output"][0]['provenance'][0]["start_paragraph_id"] == obj["output"][0]['provenance'][0]["end_paragraph_id"]
                # kb = page['text'][obj["output"][0]['provenance'][0]["start_paragraph_id"]][obj["output"][0]['provenance'][0]["start_character"]:obj["output"][0]['provenance'][0]["end_character"]]
                # print(kb)
                # print(obj["output"][0]['answer'])
                turns = obj["input"].split("\n")
                if len(turns)%2 == 0: 
                    turns = ["none"] + turns
                data.append({"id":obj['id'],"dialogue":chunks(turns,2),"query":query})
    return data

data = preproc(f"wow-train-kilt.jsonl")

with open(f'parse-valid.json', 'w') as fp:
    json.dump(data, fp, indent=4)

data = preproc(f"wow-dev-kilt.jsonl")

with open(f'parse-test.json', 'w') as fp:
    json.dump(data, fp, indent=4)



# def preproc(split):
#     dataset = []
#     cnt = 0
#     cnt_empty_meta = 0
#     cnt_empty_meta_turn = 0
#     total_turns = 0
#     for idx_d, d in tqdm(enumerate(split),total=len(split)):
#         # print(d.keys())
#         print(d['chosen_topic'], d['persona'],d["wizard_eval"] ,len(d['dialog']),len(d['chosen_topic_passage']))
#         # print(d['dialog'].keys())
#         # exit()
#         dialogue = {"meta":[d['persona']], "dialogue": [], "KB": [], "query": []}
#         if("Wizard" in d['dialog'][0]['speaker']):
#             # cnt += 1
#             d['dialog'] = d['dialog'][1:]
#         for i in range(0,len(d['dialog']),2):
#             # print(d['dialog'][i]['speaker'],len(d['dialog'][i]['retrieved_passages']))
#             # print(d['dialog'][i]['speaker'],len(d['dialog'][i]['text']))
#             # print(d['dialog'][i+1]['speaker'],len(d['dialog'][i+1]['retrieved_passages']))
#             # print(d['dialog'][i+1]['checked_sentence'],d['dialog'][i+1]['checked_passage'])
#             # print(d['dialog'][i+1]['speaker'],d['dialog'][i+1]['text'])
#             # exit()
#             if(i+1>=len(d['dialog'])): break
#             if ("Apprentice" not in d['dialog'][i]["speaker"]): 
#                 cnt += 1
#                 break
#             if ("Wizard" not in d['dialog'][i+1]["speaker"]): 
#                 cnt += 1
#                 break

#             print("DIALOGUE INPUT: ",d['dialog'][i]['text'])
#             # print("ENTITES",entity)
#             # for s in sentences:
#             #     print("WIKI",s)
#             print("DIALOGUE RESPONSE: ",d['dialog'][i+1]['text'])
#             # print()
#             # print(d['dialog'][i].keys())
#             print(d['dialog'][i+1].keys())

#             print(d['dialog'][i+1]["checked_sentence"])
#             print(d['dialog'][i+1]["checked_passage"])
#             print(d['dialog'][i+1]["retrieved_topics"])

#             print(len(d['dialog'][i+1]["retrieved_passages"]))
#             for i in range(7):
#                 print(d['dialog'][i+1]["retrieved_passages"][i].keys())
#                 print(d['dialog'][i+1]["retrieved_passages"][i])
#                 input()
#                 break
#             doc = list(d['dialog'][i+1]['checked_sentence'].values())
#             if(len(doc)>0 and doc[0]=="no_passages_used"): doc = []
#             dialogue["KB"].append(doc)
#             dialogue["dialogue"].append([d['dialog'][i]['text'],d['dialog'][i+1]['text']])
#             input()
#         if(len(dialogue['KB'])==0): cnt_empty_meta += 1
#         dataset.append(dialogue)

#         # if(idx_d%10==0):
#         #     print("TOTAL TURNS: ", total_turns)
#         #     print("NO ENTITY TURNS: ", cnt_empty_meta_turn)
#         #     print()
#         #     print("TOTAL DIALOGUE: ", idx_d)
#         #     print("NO ENTITY DIALOGUE: ", cnt_empty_meta)
#         #     with open(f"temp.json", "w", encoding="utf-8") as f:
#         #         json.dump(dataset,f,indent=4)
            
#     print(cnt)
#     return dataset


# # train = preproc(json.load(open(f"train_o.json")))
# valid = preproc(json.load(open(f"valid_random_split.json")))
# test = preproc(json.load(open(f"test_random_split.json")))

# # with open(f"train.json", "w", encoding="utf-8") as f:
# #     json.dump(train,f,indent=4)

# with open(f"valid_new.json", "w", encoding="utf-8") as f:
#     json.dump(valid,f,indent=4)

# with open(f"test_new.json", "w", encoding="utf-8") as f:
#     json.dump(test,f,indent=4)
# # print(valid[0]['dialog'][0]['text'])
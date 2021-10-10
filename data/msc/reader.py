import jsonlines
import json
from nltk.tokenize import sent_tokenize

# def preproc(split):
#     data = []
#     with jsonlines.open(f'{split}.txt') as reader:
#         for obj in reader:

#             # print(obj.keys())
#             # print()
#             # print(obj['metadata'])
#             # print("SUMMURY", summury[session_id][f"{obj['metadata']['initial_data_id']}"])
#             # print("SPEAKER 1",obj['personas'][0])
#             # print("SPEAKER 2",obj['personas'][1])
#             # print(obj['metadata'])
#             # print(obj['init_personas'])
#             dial = {"id":obj['metadata']['initial_data_id'],"meta":[],"dialogue":[]}
#             dial["meta"] = {"user":obj['personas'][0],"assistant":obj['personas'][1]}
#             temp = []
#             for d in obj['dialog']:
#                 if d['id'] == "Speaker 1":
#                     temp = [d['text']]
#                 elif d['id'] == "Speaker 2":
#                     if len(temp)==0:
#                         print(obj['metadata']['initial_data_id'])
#                     else:
#                         temp.append(d['text'])
#                         dial['dialogue'].append(temp)
#             #     # print(f"{d['id']}: {d['text']}")
#             # input()
#             data.append(dial)
#     return data

# for split in ["valid","test"]:
#     for session_id in range(2,6):
#         data = preproc(f"msc/msc_dialogue/session_{session_id}/{split}")
#         with open(f'../msc/session-{session_id}-{split}.json', 'w') as fp:
#             json.dump(data, fp, indent=4)
    # summury_session = json.load(open("msc/msc_dialogue/sessionlevel_summaries_subsample5.json"))
    # summury = json.load(open("msc/msc_dialogue/summaries_subsample5.json"))

def preproc(split):
    data = []
    with jsonlines.open(f'{split}.txt') as reader:
        for obj in reader:

            dial = {"id":obj['initial_data_id'],"user_memory":[],"dialogue":[]}
            temp = []
            for d in obj['dialog']:
                # print(f"{d['id']}: {d['text']}")
                if d['id'] == "bot_0":
                    temp = [d['text']]
                    if 'persona_text' in d:
                        # print("TLDR: ", d['persona_text'])
                        dial['user_memory'].append([d['persona_text']])
                    else:
                        # print("TLDR: None")
                        dial['user_memory'].append([])
                elif d['id'] == "bot_1":
                    temp.append(d['text'])
                    dial['dialogue'].append(temp)

            if len(dial["dialogue"]) != len(dial["user_memory"]):
                temp.append("")
                dial['dialogue'].append(temp)

            assert len(dial["dialogue"]) == len(dial["user_memory"])
            data.append(dial)
    return data

for split in ["valid","test"]:
    print(split)
    for session_id in range(1,5):
        print(session_id)
        data = preproc(f"msc/msc_personasummary/session_{session_id}/{split}")

        with open(f'../msc/parse-session-{session_id}-{split}.json', 'w') as fp:
            json.dump(data, fp, indent=4)
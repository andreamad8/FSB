import jsonlines
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

def preproc(split):
    data = []
    with jsonlines.open(f'{split}.jsonl') as reader:
        # 
        for obj in tqdm(reader):
            elm = obj[list(obj.keys())[0]]
            dial = {"id":list(obj.keys())[0],"meta":elm["apprentice_persona"].split("\n"),"dialogue":[]}
            # print("PERSONA: ",elm["apprentice_persona"])
            for turn in elm['dialog_history']:
                if turn['action'] == "SearchAgent => Wizard":
                    memory = []
                    for c in turn['context']['contents']:
                        # print(c['url'])
                        memory.append(c['content'])
                else:
                    if "selected_contents" in turn['context']:
                        for i, s in enumerate(turn['context']['selected_contents'][1:]):
                            if any(s):
                                for z in [j for j, x in enumerate(s) if x]:
                                    # print(f"KB: {''.join(sent_tokenize(memory[i][z])[:1])}")   
                                    try:
                                        dial["dialogue"].append({"action":"KB","text":''.join(sent_tokenize(memory[i][z])[:1])})
                                    except:
                                        dial["dialogue"].append({"action":"KB","text":""})

                    dial["dialogue"].append({"action":turn['action'],"text":turn['text']})
                    
                    # print(f"{turn['action']}: {turn['text']}")
            
            ## remove duplicate turns of search ==> take the last search
            new_dial = []
            for i in range(len(dial["dialogue"])):
                if (i==0) or dial["dialogue"][i]["action"] != dial["dialogue"][i-1]["action"]:
                    new_dial.append(dial["dialogue"][i])
                else:
                    new_dial[-1] = dial["dialogue"][i]

            dial["dialogue"] = new_dial


            ### convert to format
            new_dial = []
            temp_turn = []
            temp_query = []
            temp_KB = []
            i = 0
            while i<len(dial["dialogue"]):
                ### this is a special case where the first turn is done by the wizard
                if i==0 and dial["dialogue"][i]["action"] == "Wizard => SearchAgent" or dial["dialogue"][i]["action"] == "Wizard => Apprentice":

                    if dial["dialogue"][i]["action"] == "Wizard => SearchAgent":
                        temp_query.append([dial["dialogue"][i]["text"].strip()])
                        i+=1 
                    else:
                        temp_query.append([])
                    if dial["dialogue"][i]["action"] == "KB":
                        temp_KB.append([dial["dialogue"][i]["text"]]) 
                        i+=1 
                    else:
                        temp_KB.append([]) 
                    if dial["dialogue"][i]["action"] == "Wizard => Apprentice":

                        temp_turn = ["none",dial["dialogue"][i]["text"].strip()]
                        new_dial.append(temp_turn)
                        temp_turn = []
                    else:
                        print("ERROR 1")
                        input()

                else:
                    # print(dial["dialogue"][i]["action"])
                    ## i != len(dial["dialogue"])-1 ==> THIS MEANS NOT LAST TURN
                    if dial["dialogue"][i]["action"] == "Apprentice => Wizard" and i != len(dial["dialogue"])-1:
                        temp_turn.append(dial["dialogue"][i]["text"].strip())
                        if i+1>=len(dial["dialogue"]): 
                            break
                        if dial["dialogue"][i+1]["action"] == "Wizard => SearchAgent":
                            i+=1 
                            temp_query.append([dial["dialogue"][i]["text"].strip()])
                        else:
                            temp_query.append([])
                        if i+1>=len(dial["dialogue"]): 
                            temp_query = temp_query[:-1]
                            break
                        if dial["dialogue"][i+1]["action"] == "KB":
                            i+=1 
                            temp_KB.append([dial["dialogue"][i]["text"]]) 
                        else:
                            temp_KB.append([]) 
                        if i+1>=len(dial["dialogue"]): 
                            break

                        if dial["dialogue"][i+1]["action"] == "Wizard => Apprentice":
                            i+=1 
                            temp_turn.append(dial["dialogue"][i]["text"].strip())
                            new_dial.append(temp_turn)
                            temp_turn = []

                i += 1

            assert len(new_dial) == len(temp_query)
            assert len(new_dial) == len(temp_KB)
            dial["dialogue"] = new_dial
            dial["query"] = temp_query
            dial["KB"] = temp_KB
            data.append(dial)


    with open(f'{split}.json', 'w') as fp:
        json.dump(data, fp, indent=4)

# preproc("valid")
# preproc("test")
preproc("train")
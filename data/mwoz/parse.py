import json
from dictdiffer import diff


def make_data_single_domain(data,split):

    dataset = {
                "hotel":[],
                "taxi":[],
                "attraction":[],
                "train":[],
                "restaurant":[]
    }


    for dial in data:
        if "SNG" in dial['dial_id']:
            state = {}
            dial_new = {
                    "meta": [],
                    "dialogue": [],
                    "state":[],
                    "dial_id":""
                    }
            turns_in_dialogue = []
            dial_new["state"].append("none")
            for turn in dial["turns"]:
                if turn['system'] != "none":
                    turns_in_dialogue.append(turn['system'])
                turns_in_dialogue.append(turn['user'])
                diff_state = list(diff(state,turn['state']['slot_values']))
                if len(diff_state)>0:
                    dst_list = []
                    for operation in diff_state:
                        op, slot_change, diff_list = operation
                        if op in ["add","remove"]:
                            for slot, value in diff_list:
                                dst_list.append(f"{slot.replace(' ','_')}={value}")
                        else: 
                            assert op == "change"
                            dst_list.append(f"{slot_change.replace(' ','_')}={diff_list[1]}")

                    dst_string = '\t'.join(dst_list)
                    dial_new["meta"].append([dst_string])
                else:
                    dial_new["meta"].append([])

                state_string_li = []
                for slot, value in turn['state']['slot_values'].items():
                    state_string_li.append(f"{slot.replace(' ','_')}={value}")
                dial_new["state"].append("\t".join(state_string_li))
                # dial_new["state"].append(turn['state']['slot_values'])
                
                state = turn['state']['slot_values']
            turns_in_dialogue.append("none")
            for i in range(0, len(turns_in_dialogue), 2):
                dial_new["dialogue"].append([turns_in_dialogue[i],turns_in_dialogue[i+1]])
            dial_new["dial_id"] = dial['dial_id']
            assert len(dial["domains"]) == 1
            if dial["domains"][0] in dataset.keys():
                dataset[dial["domains"][0]].append(dial_new)

    for domain_name, data_set in dataset.items():  
        with open(f'{domain_name}-{split}.json', 'w') as fp:
            json.dump(data_set, fp, indent=4)


            
make_data_single_domain(json.load(open("dev_dials.json","r")),"valid")
make_data_single_domain(json.load(open("test_dials.json","r")),"test")


import json

def save_by_personality(data,split):
    dict_caption = json.load(open(f'resutls_caption.json', 'r'))
    data_new = []
    for dial in data:
        if dial["meta"]+".jpg" in dict_caption:
            dial["img"] = dict_caption[dial["meta"]+".jpg"][0]
        else:
            dial["img"] = ""
            print(dial["meta"])
        data_new.append(dial)
        # print(dial)
        # input()
    with open(f'{split}_img.json', 'w') as fp:
        json.dump(data_new, fp, indent=4)
    # print()

save_by_personality(json.load(open("test.json","r")),"test")
save_by_personality(json.load(open("valid.json","r")),"valid")
import json
import re
# read babi dataset and convert to json format
def read_babidialog(file_name):
    """
    read babi dialog dataset
    """
    babi_dialog = []
    with open(file_name, 'r') as f:
        temp = []
        KB = []
        for line in f:
            line = line.strip()
            if line == '':
                babi_dialog.append({"dialogue": temp, "KB": KB})
                temp = []
                KB = []
                continue
            if len(line.split('\t')) == 2:
                user, system = line.split('\t')
                # remove number in user utterance
                user = re.sub(r'\d+', '', user)

                temp.append([user.strip(), system.strip()])
            else:
                temp.append(["KB"])
                _, s, r, o = line.split(' ')
                KB.append([s, r, o])
    return babi_dialog

    

mapper_splt = {'trn': 'train', 'dev': 'valid', 'tst': 'test'}
for split in ['trn', 'dev', 'tst']:
    data = read_babidialog(f"dialog-babi-task6-dstc2-{split}.txt")

    # post process
    # split the dialogue by KB turns
    new_data = []
    for dialogue in data:
        first_part = {"dialogue": [], "KB": []}
        second_part = {"dialogue": [], "KB": dialogue["KB"]}
        flag = True
        for turn in dialogue['dialogue']:
            if turn[0] == "KB":
                flag = False

            if flag:
                first_part["dialogue"].append(turn)
            else:
                if turn[0] == "KB":
                    continue
                second_part["dialogue"].append(turn)
        new_data.append(first_part)
        new_data.append(second_part)

    with open(f"task6_{mapper_splt[split]}.json", 'w') as f:
        json.dump(new_data, f, indent=4)

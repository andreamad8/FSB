import json
import re
import random
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
                babi_dialog.append({"dialogue": temp, "meta": KB})
                temp = []
                KB = []
                continue
            if len(line.split('\t')) == 2:
                user, system = line.split('\t')
                # remove number in user utterance
                user = re.sub(r'\d+', '', user)

                temp.append([user.strip(), system.strip()])
            else:
                temp.append(["meta"])
                _, s, r, o = line.split(' ')
                KB.append([s, r, o])
    return babi_dialog

    


mapper_splt = {'trn': 'train', 'dev': 'valid', 'tst': 'test'}
for name_base in ["dialog-babi-task5-full-dialogs"]:
    for split in ['trn', 'dev', 'tst']:
        data = read_babidialog(f"{name_base}-{split}.txt")

        # post process
        # split the dialogue by KB turns
        new_data = []
        new_data_first = []
        new_data_second = []
        for dialogue in data:
            # generate a random alpha numberic id for each dialogue
            dialogue_id = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
            first_part = {"id":dialogue_id, "dialogue": [], "meta": []}
            second_part = {"id":dialogue_id, "dialogue": [], "meta": dialogue["meta"]}
            flag = True
            for turn in dialogue['dialogue']:
                if turn[0] == "meta":
                    flag = False

                if flag:
                    first_part["dialogue"].append(turn)
                else:
                    if turn[0] == "meta":
                        continue
                    second_part["dialogue"].append(turn)
            new_data.append(first_part)
            new_data.append(second_part)
            new_data_first.append(first_part)
            new_data_second.append(second_part)


        if "5" in name_base: num = 5
        else: num = 6
        with open(f"bAbI-dial-{num}-{mapper_splt[split]}.json", 'w') as f:
            json.dump(new_data, f, indent=4)
        with open(f"bAbI-dial-{num}-first-{mapper_splt[split]}.json", 'w') as f:
            json.dump(new_data_first, f, indent=4)
        with open(f"bAbI-dial-{num}-second-{mapper_splt[split]}.json", 'w') as f:
            json.dump(new_data_second, f, indent=4)

data = read_babidialog(f"dialog-babi-task5-full-dialogs-tst-OOV.txt")

# post process
# split the dialogue by KB turns
new_data = []
new_data_first = []
new_data_second = []
for dialogue in data:
    dialogue_id = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
    first_part = {"id":dialogue_id, "dialogue": [], "meta": []}
    second_part = {"id":dialogue_id, "dialogue": [], "meta": dialogue["meta"]}
    flag = True
    for turn in dialogue['dialogue']:
        if turn[0] == "meta":
            flag = False

        if flag:
            first_part["dialogue"].append(turn)
        else:
            if turn[0] == "meta":
                continue
            second_part["dialogue"].append(turn)
    new_data.append(first_part)
    new_data.append(second_part)
    new_data_first.append(first_part)
    new_data_second.append(second_part)

with open(f"bAbI-dial-5-OOV-test.json", 'w') as f:
    json.dump(new_data, f, indent=4)
with open(f"bAbI-dial-5-OOV-first-test.json", 'w') as f:
    json.dump(new_data_first, f, indent=4)
with open(f"bAbI-dial-5-OOV-second-test.json", 'w') as f:
    json.dump(new_data_second, f, indent=4)

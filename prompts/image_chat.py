def convert_sample_to_shot_IC_prefix(sample, with_knowledge=None):
    prefix = "Dialogue:\n"
    for turn, personality in zip(sample["dialogue"],sample["personalities"]):
        prefix += f"{personality[0]}: {turn[0]}" +"\n"
        if turn[1] != "" :
            prefix += f"{personality[1]}: {turn[1]}" +"\n"

    return prefix

def convert_sample_to_shot_IC_inference(sample, with_knowledge=None):
    prefix = "Dialogue:\n"
    assert len(sample["dialogue"]) == len(sample["personalities"])
    for turn, personality in zip(sample["dialogue"],sample["personalities"]):
        
        if turn[0] != "":
            prefix += f"{personality[0]}: {turn[0]}" +"\n"
        else:
            prefix += f"{personality[0]}:"
            return prefix

        if turn[1] != "" :
            prefix += f"{personality[1]}: {turn[1]}" +"\n"
        else:
            prefix += f"{personality[1]}:"
            return prefix

    return prefix


def convert_sample_to_shot_IC_prefix_interact(sample, with_knowledge=None):
    prefix = "Dialogue:\n"
    prefix += f"System: {sample['dialogue'][0][0]}" +"\n"
    prefix += f"User: {sample['dialogue'][0][1]}" +"\n"
    style = sample['personalities'][1][0].replace(" ","-").replace("(","").replace(")","").replace(",","").split("_")[0]
    prefix += f"{style}: {sample['dialogue'][1][0]}" +"\n"
    return prefix


def convert_sample_to_shot_IC_interact(sample, with_knowledge=None):
    prefix = "Dialogue:\n"
    # assert len(sample["dialogue"]) == len(sample["personalities"])
    for turn in sample["dialogue"]:
        
        prefix += f"User: {turn[0]}" +"\n"

        if turn[1] != "" :
            prefix += f"System: {turn[1]}" +"\n"
        else:
            style = sample['personalities'].replace(" ","-").replace("(","").replace(")","").replace(",","").split("_")[0]
            prefix += f"{style}:"
            return prefix

    return prefix

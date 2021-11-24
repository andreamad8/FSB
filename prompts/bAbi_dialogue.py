def convert_sample_to_shot_bAbi(sample, with_knowledge=None):
    prefix = "KB:\n"
    if len(sample["meta"])==0:
        prefix += "empty\n"
    else:
        for s in sample["meta"]:
            prefix += "\t".join(s)+"\n"

    prefix += "Dialogue\n"
    for turn in sample["dialogue"]:
        prefix += f"U: {turn[0]}" +"\n"
        if turn[1] == "":
            prefix += f"S:" 
            return prefix
        else:
            prefix += f"S: {turn[1]}" +"\n"

    return prefix



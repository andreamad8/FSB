def convert_sample_to_shot_smd(sample, with_knowledge=None):
    prefix = "KB:\n"
    for s in sample["meta"]:
        prefix += s+"\n"

    prefix += "Dialogue:\n"
    for turn in sample["dialogue"]:
        prefix += f"User: {turn[0]}" +"\n"
        if turn[1] == "":
            prefix += f"Assistant:" 
            return prefix
        else:
            prefix += f"Assistant: {turn[1]}" +"\n"

    return prefix


def convert_sample_to_shot_smd_custum(sample, with_knowledge=None):
    prefix = "KB:\n"
    for s in sample["meta"]:
        prefix += s+"\n"

    for dialogue_shot in sample["dialogue"][:with_knowledge]:
        prefix += "Dialogue:\n"
        for turn in dialogue_shot:
            prefix += f"User: {turn[0]}" +"\n"
            if turn[1] == "":
                prefix += f"Assistant:" 
                return prefix
            else:
                prefix += f"Assistant: {turn[1]}" +"\n"
        prefix += "\n\n"

    return prefix

def convert_sample_to_shot_msc(sample, with_knowledge=None):
    prefix = "User Information:\n"
    for s in sample["meta"]["user"]:
        prefix += s+"\n"

    prefix += "Assistant Information:\n"
    for s in sample["meta"]["assistant"]:
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


def convert_sample_to_shot_msc_interact(sample, with_knowledge=None):
    prefix = "User Information:\n"
    for s in sample["user"]:
        prefix += s+"\n"

    prefix += "Assistant Information:\n"
    for s in sample["assistant"]:
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

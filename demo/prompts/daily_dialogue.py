def convert_sample_to_shot_DD_prefix(sample, with_knowledge=None):
    prefix = "Dialogue:\n"
    for turn in sample["dialogue"]:
        prefix += f"User A: {turn[0]}" +"\n"
        if turn[1] != "" :
            prefix += f"User B: {turn[1]}" +"\n"

    return prefix

def convert_sample_to_shot_DD_inference(sample, with_knowledge=None):
    prefix = "Dialogue:\n"
    for turn in sample["dialogue"]:
        if turn[0] != "":
            prefix += f"User A: {turn[0]}" +"\n"
        else:
            prefix += f"User A:"
            return prefix

        if turn[1] != "" :
            prefix += f"User B: {turn[1]}" +"\n"
        else:
            prefix += f"User B:"
            return prefix

    return prefix

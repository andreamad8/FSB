def convert_sample_to_shot_wow(sample, level=None):
    prefix = "Dialogue:\n"
    for id_t, turn in enumerate(sample["dialogue"]):
        if len(turn) == 2:
            prefix += f"User: {turn[0]}" +"\n"
            prefix += f"Assistant: {turn[1]}" +"\n"
        else:
            prefix += f"User: {turn[0]}" +"\n"

    if sample["query"] == "":
        prefix += f"Search:"
    else:
        prefix += f"Search: {sample['query']}" 

    return prefix

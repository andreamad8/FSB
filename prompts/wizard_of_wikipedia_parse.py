def convert_sample_to_shot_wow(sample, level):
    prefix = "Dialogue:\n"
    for id_t, turn in enumerate(sample["dialogue"]):
        if id_t == len(sample["dialogue"])-1:
            prefix += f"User: {turn[0]}" +"\n"
            if turn[1] != "":
                prefix += f"Search: {turn[1]}" +"\n"
            else:
                prefix += f"Search:" 
                return prefix
        else:
            prefix += f"User: {turn[0]}" +"\n"
            prefix += f"Assistant: {turn[1]}" +"\n"

    return prefix

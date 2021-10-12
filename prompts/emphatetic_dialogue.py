def convert_sample_to_shot_ed(sample, with_knowledge=None):
    prefix = "Dialogue:\n"
    for turn in sample["dialogue"]:
        prefix += f"User: {turn[0]}" +"\n"
        if turn[1] == "":
            prefix += f"Empath:" 
            return prefix
        else:
            prefix += f"Empath: {turn[1]}" +"\n"
    return prefix

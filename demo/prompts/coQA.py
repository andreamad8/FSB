def convert_sample_to_shot_coQA(sample, with_knowledge=None):
    prefix = f"{sample['meta']}\n"

    for turn in sample["dialogue"]:
        prefix += f"Q: {turn[0]}" +"\n"
        if turn[1] == "":
            prefix += f"A:" 
            return prefix
        else:
            prefix += f"A: {turn[1]}" +"\n"

    return prefix



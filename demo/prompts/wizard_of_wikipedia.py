def convert_sample_to_shot_wow(sample, with_knowledge=True):
    prefix = "Dialogue:\n"
    assert len(sample["dialogue"]) == len(sample["meta"])
    for turn, meta in zip(sample["dialogue"],sample["meta"]):
        prefix += f"User: {turn[0]}" +"\n"
        if with_knowledge:
            if len(meta)>0:
                prefix += f"KB: {meta[0]}" +"\n"
            else: 
                prefix += f"KB: None" +"\n"
        if turn[1] == "":
            prefix += f"Assistant:" 
            return prefix
        else:
            prefix += f"Assistant: {turn[1]}" +"\n"

    return prefix

def convert_sample_to_shot_wow_interact(sample, with_knowledge=True):
    prefix = "Dialogue:\n"
    assert len(sample["dialogue"]) == len(sample["KB_wiki"])
    for turn, meta in zip(sample["dialogue"],sample["KB_wiki"]):
        prefix += f"User: {turn[0]}" +"\n"
        if with_knowledge:
            if len(meta)>0:
                prefix += f"KB: {meta[0]}" +"\n"
            else: 
                prefix += f"KB: None" +"\n"
        if turn[1] == "":
            prefix += f"Assistant:" 
            return prefix
        else:
            prefix += f"Assistant: {turn[1]}" +"\n"

    return prefix

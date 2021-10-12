def convert_sample_to_shot_dialKG(sample,with_knowledge):
    prefix = "Dialogue:\n"
    assert len(sample["dialogue"]) == len(sample["KG"])
    for turn, meta in zip(sample["dialogue"],sample["KG"]):
        prefix += f"User: {turn[0]}" +"\n"
        if with_knowledge and len(meta)>0:
            prefix += f"KG: {meta[0]}" +"\n"
        if turn[1] == "":
            prefix += f"Assistant:" 
            return prefix
        else:
            prefix += f"Assistant: {turn[1]}" +"\n"
            
    return prefix


def convert_sample_to_shot_dialKG_interact(sample,with_knowledge):
    prefix = "Dialogue:\n"
    assert len(sample["dialogue"]) == len(sample["KG"])
    for turn, meta in zip(sample["dialogue"],sample["KG"]):
        prefix += f"User: {turn[0]}" +"\n"
        if with_knowledge and len(meta)>0:
            prefix += f"KG: {meta[0]}" +"\n"
        if turn[1] == "":
            prefix += f"Assistant:" 
            return prefix
        else:
            prefix += f"Assistant: {turn[1]}" +"\n"
            
    return prefix

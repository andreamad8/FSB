def convert_sample_to_shot_msc(sample, level=None):
    prefix = "Dialogue:\n"
    assert len(sample["dialogue"]) == len(sample["user_memory"])
    for turn, meta in zip(sample["dialogue"],sample["user_memory"]):
        prefix += f"User: {turn[0]}" +"\n"
        if turn[1] != "":
            if len(meta)>0:
                prefix += f"Write: {meta[0]}" +"\n"
            else:
                prefix += f"Write: None" +"\n"
        if turn[1] == "":
            prefix += f"Write:" 
            return prefix
        else:
            prefix += f"Persona: {turn[1]}" +"\n"
            
    return prefix

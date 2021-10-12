def convert_sample_to_shot_dialKG(sample, level):
    prefix = "Dialogue:\n"
    assert len(sample["dialogue"]) == len(sample["KG"])
    for turn, meta in zip(sample["dialogue"],sample["KG"]):
        prefix += f"User: {turn[0]}" +"\n"
        if turn[1] != "":
            if len(meta)>0:
                prefix += f"KG: {meta[0]}" +"\n"
            else:
                prefix += f"KG: None" +"\n"
        if turn[1] == "":
            prefix += f"KG:" 
            return prefix
        else:
            prefix += f"Assistant: {turn[1]}" +"\n"
            
    return prefix

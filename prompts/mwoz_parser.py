def convert_sample_to_shot_mwoz(sample, level):
    prefix = "Dialogue:\n"
    # print(len(sample["dialogue"]),len(sample["meta"]))
    assert len(sample["dialogue"]) == len(sample["meta"])
    for turn, meta in zip(sample["dialogue"],sample["meta"]):
        prefix += f"User: {turn[0]}" +"\n"
        if turn[1] != "":
            if len(meta)>0:
                prefix += f"DST: {meta[0]}" +"\n"
            else:
                prefix += f"DST: None" +"\n"
        if turn[1] == "":
            prefix += f"DST:" 
            return prefix
        else:
            prefix += f"Assistant: {turn[1]}" +"\n"
            
    return prefix

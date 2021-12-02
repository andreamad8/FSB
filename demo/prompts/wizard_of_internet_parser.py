def convert_sample_to_shot_wit(sample, level):
    '''
        {
        "meta": [List[str]],
        "dialogue": [
                    ["str:User","str:Sys"]
                    ]
        }
    '''


    prefix = "Assistant Information:\n"
    for s in sample["meta"]:
        prefix += s+"\n"

    prefix += "Dialogue:\n"
    assert len(sample["dialogue"]) == len(sample["query"])
    for turn, meta in zip(sample["dialogue"],sample["query"]):
        prefix += f"User: {turn[0]}" +"\n"
        if turn[1] != "":
            if len(meta)>0:
                prefix += f"Search: {meta[0]}" +"\n"
            else:
                prefix += f"Search: None" +"\n"
        if turn[1] == "":
            prefix += f"Search:" 
            return prefix
        else:
            prefix += f"Assistant: {turn[1]}" +"\n"
            
    return prefix

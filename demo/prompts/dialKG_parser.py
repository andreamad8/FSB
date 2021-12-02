def convert_sample_to_shot_dialKG(sample, level=None):
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
        if len(sample["query"]) == 1:
            query = "\t".join(sample["query"][0])
        else:
            query = "\t".join(sample["query"][0]) + "\t\t" + "\t".join(sample["query"][1]) 
        prefix += f"Search: {query}" 

    return prefix

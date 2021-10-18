def convert_sample_to_shot_selector(sample, with_knowledge=None):
    '''
        {
        "meta": [List[str]],
        "dialogue": [
                    ["str:User","str:Sys"]
                    ]
        }
    '''

    prefix = "Dialogue:\n"
    for turn in sample["dialogue"]:
        prefix += f"User: {turn[0]}" +"\n"
        if turn[1] == "":
            ## NO GENERATION REQUIRED
            pass
        else:
            prefix += f"Assistant: {turn[1]}" +"\n"
    return prefix

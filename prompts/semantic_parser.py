def convert_sample_to_shot_semantic_parser(sample, with_knowledge=None):
    '''
        {
        "meta": [List[str]],
        "dialogue": [
                    ["str:User","str:Sys"]
                    ]
        }
    '''

    if sample["query"] != "":
        return f"User: {sample['dialogue']}\nQuery: {sample['query']}"
    else:
        return f"User: {sample['dialogue']}\nQuery:"


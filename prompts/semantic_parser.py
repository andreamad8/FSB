def convert_sample_to_shot_semantic_parser(sample, level=None):
    if sample["query"] != "":
        return f"{sample['dialogue']} {sample['query']}"
    else:
        return f"{sample['dialogue']}"


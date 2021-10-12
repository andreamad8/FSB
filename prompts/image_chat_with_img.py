def convert_sample_to_shot_IC_img_prefix(sample, with_knowledge=None):
    prefix = f"Image: {sample['img']}\n"
    prefix += "Dialogue:\n"
    for turn, personality in zip(sample["dialogue"],sample["personalities"]):
        prefix += f"{personality[0]}: {turn[0]}" +"\n"
        if turn[1] != "" :
            prefix += f"{personality[1]}: {turn[1]}" +"\n"

    return prefix

def convert_sample_to_shot_IC_img_inference(sample, with_knowledge=None):
    prefix = f"Image: {sample['img']}\n"
    prefix += "Dialogue:\n"
    assert len(sample["dialogue"]) == len(sample["personalities"])
    for turn, personality in zip(sample["dialogue"],sample["personalities"]):
        
        if turn[0] != "":
            prefix += f"{personality[0]}: {turn[0]}" +"\n"
        else:
            prefix += f"{personality[0]}:"
            return prefix

        if turn[1] != "" :
            prefix += f"{personality[1]}: {turn[1]}" +"\n"
        else:
            prefix += f"{personality[1]}:"
            return prefix

    return prefix


from collections import defaultdict

def convert_sample_to_shot_bAbi(sample, with_knowledge=None):
    prefix = ""
    if len(sample["meta"])!=0:
        dict_restaurant = defaultdict(list)
        for s in sample["meta"]:
            dict_restaurant[s[0]].append([s[1], s[2]])
        
        prefix = "KB\n"
        for key, value in dict_restaurant.items():
            prefix += "R_name: "+ key + "\t"
            for v in value:
                prefix += v[0] + ": " + v[1] + "\t"
            prefix += "\n"
            # prefix += "\t".join(s)+"\n"
            # prefix += key+"\n"

    prefix += "Dialogue\n"
    for turn in sample["dialogue"]:
        prefix += f"U: {turn[0]}" +"\n"
        if turn[1] == "":
            prefix += f"S:" 
            return prefix
        else:
            prefix += f"S: {turn[1]}" +"\n"

    return prefix



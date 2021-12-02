def convert_sample_to_shot_persona(sample, with_knowledge=None):
    prefix = "Persona Information:\n"
    for s in sample["meta"]:
        prefix += s+"\n"

    prefix += "Dialogue:\n"
    for turn in sample["dialogue"]:
        prefix += f"User: {turn[0]}" +"\n"
        if turn[1] == "":
            prefix += f"Persona:" 
            return prefix
        else:
            prefix += f"Persona: {turn[1]}" +"\n"

    return prefix



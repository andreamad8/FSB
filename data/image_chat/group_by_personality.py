import json



personality = ["Adventurous", "Appreciative (Grateful)",
 "Articulate (Well-spoken, Expressive)", "Attractive", "Brilliant", 
 "Calm", "Captivating", "Caring", "Charming", "Cheerful", "Clever", 
 "Colorful (Full of Life, Interesting)", "Compassionate (Sympathetic, Warm)",
 "Confident", "Considerate", "Contemplative (Reflective, Thoughtful)", 
 "Courageous", "Creative", "Cultured (Refined, Educated)", "Curious", 
 "Daring", "Deep", "Dramatic", "Earnest (Enthusiastic)", "Elegant", 
 "Eloquent (Well-spoken, Expressive)", "Empathetic", "Energetic", 
 "Enthusiastic", "Exciting", "Extraordinary", "Freethinking", 
 "Fun-loving", "Gentle", "Happy", "Honest", "Humble", "Humorous", 
 "Idealistic", "Imaginative", "Insightful", "Intelligent", "Kind", 
 "Knowledgeable", "Logical", "Meticulous (Precise, Thorough)", 
 "Objective (Detached, Impartial)", "Observant", "Open", 
 "Optimistic", "Passionate", "Patriotic", "Peaceful", 
 "Perceptive", "Playful", "Practical", "Profound", "Rational", 
 "Realistic", "Reflective", "Relaxed", "Respectful", "Romantic", 
 "Rustic (Rural)", "Scholarly", "Sensitive", "Sentimental", 
 "Serious", "Simple", "Sophisticated", "Spirited", "Spontaneous", 
 "Stoic (Unemotional, Matter-of-fact)", "Suave (Charming, Smooth)", 
 "Sweet", "Sympathetic", "Vivacious (Lively, Animated)", "Warm", 
 "Wise", "Witty", "Youthful", "Absentminded", 
 "Aggressive", "Amusing", "Artful", "Boyish", "Breezy (Relaxed, Informal)", 
 "Businesslike", "Casual", "Cerebral (Intellectual, Logical)", 
 "Complex", "Conservative (Traditional, Conventional)", 
 "Contradictory", "Cute", "Dreamy", "Dry", "Emotional", 
 "Enigmatic (Cryptic, Obscure)", "Formal", "Glamorous", 
 "High-spirited", "Impersonal", "Intense", "Maternal (Mother-like)", 
 "Mellow (Soothing, Sweet)", "Mystical", "Neutral", "Old-fashioned", 
 "Ordinary", "Questioning", "Sarcastic", "Sensual", "Skeptical", 
 "Solemn", "Stylish", "Tough", "Whimsical (Playful, Fanciful)", 
 "Abrasive (Annoying, Irritating)", 
 "Airy (Casual, Not Serious)", "Aloof (Detached, Distant)", 
 "Angry", "Anxious", "Apathetic (Uncaring, Disinterested)", 
 "Argumentative", "Arrogant", "Artificial", "Assertive", 
 "Barbaric", "Bewildered (Astonished, Confused)", "Bizarre", 
 "Bland", "Blunt", "Boisterous (Rowdy, Loud)", "Childish", 
 "Coarse (Not Fine, Crass)", "Cold", "Conceited (Arrogant, Egotistical)", 
 "Confused", "Contemptible (Despicable, Vile)", "Cowardly", "Crazy", 
 "Critical", "Cruel", "Cynical (Doubtful, Skeptical)", "Destructive", 
 "Devious", "Discouraging", "Disturbing", "Dull", "Egocentric (Self-centered)", 
 "Envious", "Erratic", "Escapist (Dreamer, Seeks Distraction)", "Excitable", 
 "Extravagant", "Extreme", "Fanatical", "Fanciful", "Fatalistic (Bleak, Gloomy)", 
 "Fawning (Flattering, Deferential)", "Fearful", "Fickle (Changeable, Temperamental)", 
 "Fiery", "Foolish", "Frightening", "Frivolous (Trivial, Silly)", "Gloomy", 
 "Grand", "Grim", "Hateful", "Haughty (Arrogant, Snobbish)", "Hostile", 
 "Irrational", "Irritable", "Lazy", "Malicious", "Melancholic", "Miserable", 
 "Money-minded", "Monstrous", "Moody", "Morbid", "Narcissistic (Self-centered, Egotistical)", 
 "Neurotic (Manic, Obsessive)", "Nihilistic", "Obnoxious", "Obsessive", 
 "Odd", "Offhand", "Opinionated", "Outrageous", "Overimaginative",
 "Paranoid", "Passive", "Pompous (Self-important, Arrogant)", 
 "Pretentious (Snobbish, Showy)", "Provocative", "Quirky", 
 "Resentful", "Ridiculous", "Rigid", "Rowdy", "Scornful", 
 "Shy", "Silly", "Stiff", "Stupid", "Tense", "Uncreative", 
 "Unimaginative", "Unrealistic", "Vacuous (Empty, Unintelligent)", 
 "Vague", "Wishful", "Zany"]


def save_by_personality(data,split):
    dial_by_personality_turn_1 = {p:[] for p in personality}
    dial_by_personality_turn_2 = {p:[] for p in personality}
    for dial in data:
        if dial["personalities"][0][1] not in dial_by_personality_turn_1:
            print(dial["personalities"][0][1])
            print("MISSING")
        else:
            dial_by_personality_turn_1[dial["personalities"][0][1]].append(dial)

        if dial["personalities"][1][0] not in dial_by_personality_turn_2:
            print(dial["personalities"][1][0])
            print("MISSING")
        else:
            dial_by_personality_turn_2[dial["personalities"][1][0]].append(dial)


    for k,v in dial_by_personality_turn_1.items():
        # if len(v) < 40:
        name = k.replace(" ","-").replace("(","").replace(")","").replace(",","")
        with open(f'{split}/{name}_1.json', 'w') as fp:
            json.dump(v, fp, indent=4)
    for k,v in dial_by_personality_turn_2.items():
        # if len(v) < 40:
        name = k.replace(" ","-").replace("(","").replace(")","").replace(",","")
        with open(f'{split}/{name}_2.json', 'w') as fp:
            json.dump(v, fp, indent=4)
    # print()

save_by_personality(json.load(open("test.json","r")),"test")
save_by_personality(json.load(open("valid.json","r")),"valid")

# for k,v in dial_by_personality_turn_2.items():
#     if len(v) < 20:
#         print(k, len(v))
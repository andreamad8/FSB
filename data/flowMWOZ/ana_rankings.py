import glob
import json
from collections import defaultdict

for file in glob.glob("test_dynamic*.json"):
    
    stats = defaultdict(list)
    avg_rank = list()
    with open(file) as f:
        data = json.load(f)
        for d in data:
            for i in range(len(d["shots"])):
                if "all-mpnet" in file:
                    stats[i].append(d["shots"][i]["score"])
                else:
                    stats[i].append(d["shots"][i]["tfidf"])
            temp = 0
            for i in range(len(d["shots"])):
                if "all-mpnet" in file:
                    temp += d["shots"][i]["score"]
                else:
                    temp += d["shots"][i]["tfidf"]
            avg_rank.append(temp/len(d["shots"]))
    print("File name is: ", file)
    print("Average rank is: ", sum(avg_rank)/len(avg_rank))
    for i in range(len(stats)):
        print("Average score for shot", i, "is: ", sum(stats[i])/len(stats[i]))
    input("Press Enter to continue...")

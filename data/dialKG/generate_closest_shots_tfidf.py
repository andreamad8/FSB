import csv
import json
import numpy as np
import pandas as pd
import glob
import random
# https://github.com/asvskartheek/Text-Retrieval/blob/master/TF-IDF%20Search%20Engine%20(SKLEARN).ipynb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def flatten_dialogue(dialogue):
    """
    Flatten a dialogue into a single string
    Dialogue are list of turns (list of lists)
    """
    flattened = []
    for turn in dialogue:
        # filter first element of turn if it is none
        if len(turn) == 2:
            flattened.append(" ".join(turn))
        else:
            flattened.append(turn[0])
    return " ".join(flattened)

# Get tf-idf matrix using fit_transform function
vectorizer = TfidfVectorizer()
data_dialogue_raw = []
with open("parse-train.json") as json_file:
    data_all = json.load(json_file)

# sub sample 1% 10%, 25% and all of the data 
for p in [0.01, 0.1, 0.25, 1.0]:
    # sample 3 times to get a more representative sample
    for s in range(1):
        sub_data_dialogue_raw = random.sample(data_all, int(len(data_all)*p))
        data_to_fit = [flatten_dialogue(d['dialogue']) for d in sub_data_dialogue_raw]
        print(len(data_to_fit))
        X = vectorizer.fit_transform(data_to_fit) # Store tf-idf representations of all docs

        print(X.shape) # (Number of songs, Number of unique words)

        data_temp = []
        with open("parse-test.json") as json_file:
            data = json.load(json_file)
            for i in range(len(data)):
                query_vec = vectorizer.transform([flatten_dialogue(data[i]['dialogue'])]) # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
                results = cosine_similarity(X,query_vec).reshape((-1,)) # Op -- (n_docs,1) -- Cosine Sim with each doc
                top_shots = []
                # Print Top 10 results
                for j in results.argsort()[-20:][::-1]:
                    top_shots.append({"score":results[j],
                                      "dialogue":sub_data_dialogue_raw[j]['dialogue'],
                                      "query":sub_data_dialogue_raw[j]['query']})
                data_temp.append({"dialogue":data[i]['dialogue'],"query":data[i]['query'],
                                  "id":data[i]['id'], "shots":top_shots})

        with open(f'test_dynamic_tfidf_{p}_{s}.json', 'w') as fp:
            json.dump(data_temp, fp, indent=4)



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

# Get tf-idf matrix using fit_transform function
vectorizer = TfidfVectorizer()
data_dialogue_raw = []
for file in glob.glob("train.json"):
    with open(file) as json_file:
        data_all = json.load(json_file)

# sub sample 1% 10%, 25% and all of the data 
for p in [0.01, 0.1, 0.25, 1.0]:
    # sample 3 times to get a more representative sample
    for s in range(3):
        sub_data_dialogue_raw = random.sample(data_all, int(len(data_all)*p))
        data_to_fit = [d['dialogue'] for d in sub_data_dialogue_raw]
        print(len(data_to_fit))
        X = vectorizer.fit_transform(data_to_fit) # Store tf-idf representations of all docs

        print(X.shape) # (Number of songs, Number of unique words)

        data_temp = []
        with open("test.json") as json_file:
            data = json.load(json_file)
            for i in range(len(data)):
                query_vec = vectorizer.transform([data[i]['dialogue']]) # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
                results = cosine_similarity(X,query_vec).reshape((-1,)) # Op -- (n_docs,1) -- Cosine Sim with each doc
                top_shots = []
                # Print Top 10 results
                for j in results.argsort()[-10:][::-1]:
                    top_shots.append({"tfidf":results[j],
                                      "dialogue":sub_data_dialogue_raw[j]['dialogue'],
                                      "query":sub_data_dialogue_raw[j]['query']})
                data_temp.append({"dialogue":data[i]['dialogue'],
                            "query":data[i]['query'],
                            "dialogue_id":data[i]['dialogue_id'],
                            "turn_index":data[i]['turn_index'],"shots":top_shots})

        with open(f'test_dynamic_tfidf_{p}_{s}.json', 'w') as fp:
            json.dump(data_temp, fp, indent=4)


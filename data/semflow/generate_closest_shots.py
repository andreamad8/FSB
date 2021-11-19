import csv
import json
import numpy as np
import pandas as pd
import glob
# https://github.com/asvskartheek/Text-Retrieval/blob/master/TF-IDF%20Search%20Engine%20(SKLEARN).ipynb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get tf-idf matrix using fit_transform function
vectorizer = TfidfVectorizer()
data_dialogue_raw = []
data_query = []
for file in glob.glob("train_*.json"):
    with open(file) as json_file:
        data = json.load(json_file)
        for i in range(len(data)):
            data_dialogue_raw.append(data[i]['dialogue'])
            data_query.append(data[i]['query'])

X = vectorizer.fit_transform(data_dialogue_raw) # Store tf-idf representations of all docs

print(X.shape) # (Number of songs, Number of unique words)

data_temp = []
with open("valid.json") as json_file:
    data = json.load(json_file)
    for i in range(len(data)):
        query_vec = vectorizer.transform([data[i]['dialogue']]) # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
        results = cosine_similarity(X,query_vec).reshape((-1,)) # Op -- (n_docs,1) -- Cosine Sim with each doc
        top_shots = []
        for j in results.argsort()[-30:][::-1]:
            top_shots.append({"tfidf":results[j],"dialogue":data_dialogue_raw[j],"query":data_query[j]})
        
        # print some values
        # print("Dialogue", data[i]['dialogue'])
        # print("Query", data[i]['query'])
        # print("Top 5 shots")
        # for j in range(5):
        #     print(top_shots[j]['dialogue'])
        #     print(top_shots[j]['query'])
        #     print("")
        # input()
        data_temp.append({"dialogue":data[i]['dialogue'],
                    "query":data[i]['query'],
                    "dialogue_id":data[i]['dialogue_id'],
                    "turn_index":data[i]['turn_index'],"shots":top_shots})

with open(f'test_dynamic.json', 'w') as fp:
    json.dump(data_temp, fp, indent=4)



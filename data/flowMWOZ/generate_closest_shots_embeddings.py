import csv
import json
import numpy as np
import pandas as pd
import glob
import random
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer('all-mpnet-base-v2')

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
        corpus_embeddings = embedder.encode(data_to_fit, convert_to_tensor=True)

        print(corpus_embeddings.shape) # (Number of songs, Number of unique words)

        data_temp = []
        with open("test.json") as json_file:
            data = json.load(json_file)
            for i in tqdm(range(len(data))):
                query_embedding = embedder.encode([data[i]['dialogue']], convert_to_tensor=True)
                
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=10)
                hits = hits[0]      #Get the hits for the first query
                top_shots = []
                for hit in hits:
                    top_shots.append({"score":hit['score'],
                                      "dialogue":sub_data_dialogue_raw[hit['corpus_id']]['dialogue'],
                                      "query":sub_data_dialogue_raw[hit['corpus_id']]['query']})
                data_temp.append({"dialogue":data[i]['dialogue'],
                            "query":data[i]['query'],
                            "dialogue_id":data[i]['dialogue_id'],
                            "turn_index":data[i]['turn_index'],"shots":top_shots})

        with open(f'test_dynamic_all-mpnet-base-v2_{p}_{s}.json', 'w') as fp:
            json.dump(data_temp, fp, indent=4)



import json
import jsonlines
from data.wow.KILT.kilt.knowledge_source import KnowledgeSource
from data.wow.KILT.eval_retrieval import evaluate
import requests
from tqdm import tqdm
import multiprocessing as mp
import glob


def _get_pageid_from_api(title_dict):
    pageid = 000
    if title_dict["title"] == "":
        return title_dict
    title_html = title_dict["title"].strip().replace(" ", "%20")
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&titles={}&format=json".format(
            title_html
        )
    )

    try:
        # Package the request, send the request and catch the response: r
        r = requests.get(url)

        # Decode the JSON data into a dictionary: json_data
        json_data = r.json()

        if len(json_data["query"]["pages"]) > 1:
            print("WARNING: more than one result returned from wikipedia api")

        for _, v in json_data["query"]["pages"].items():
            pageid = v["pageid"]

    except Exception as e:
        pageid = 000
        # print("Exception: {}".format(e))
    title_dict["wikipedia_id"] = pageid
    return title_dict



def score_file(file):
    pred = json.load(open(file,"r"))
    gold = json.load(open("data/wow/parse-test.json","r"))
    if 'Rprec' in pred["score"]: 
        print("Skip FILE")
        return 
    prediction = []
    for g, p in zip(gold, pred["generation"]):
        # if p["meta"][0][0] == '':
        #     prediction.append({"id":g['id'],"title":"","wikipedia_id":000})
        # else:
        #     prediction.append({"id":g['id'],"title":p["meta"][0][0].upper()[0] + p["meta"][0][0].lower()[1:],"wikipedia_id":000})
        if p["query"] == '':
            prediction.append({"id":g['id'],"title":"","wikipedia_id":000})
        else:
            prediction.append({"id":g['id'],"title":p["query"].upper()[0] + p["query"].lower()[1:],"wikipedia_id":000})
    pool = mp.Pool(mp.cpu_count())
    prediction = pool.map(_get_pageid_from_api,prediction)

    prediction_dic = {}
    for p in prediction:
        prediction_dic[p["id"]] = p
    results_list = []
    with jsonlines.open("data/wow/wow-dev-kilt.jsonl") as reader:
        for obj in reader:
            obj['output'][0]['provenance'][0]['wikipedia_id'] = prediction_dic[obj['id']]["wikipedia_id"]
            results_list.append(obj)

    score = evaluate("data/wow/wow-dev-kilt.jsonl", results_list, ks=[1,5,10,20],rank_keys=["wikipedia_id"])
    # print(score['Rprec'])
    pred["score"]['Rprec'] = score['Rprec']*100
    with open(file, 'w') as fp:
        json.dump(pred, fp, indent=4)


if __name__ == "__main__":

    for file in tqdm(glob.glob("generations/wow-parse_*.json")):
        print(file)
        score_file(file)

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import torch
import json
import os
import argparse
import random
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate
from collections import defaultdict
import os
from tqdm import tqdm
import glob
import copy
import pprint
pp = pprint.PrettyPrinter(indent=4)
import jsonlines
from datasets import Dataset



def convert_sample_to_shot_selector(sample, with_knowledge=None):
    '''
        {
        "meta": [List[str]],
        "dialogue": [
                    ["str:User","str:Sys"]
                    ]
        }
    '''

    prefix = ""
    for turn in sample["dialogue"]:
        prefix += f"User: {turn[0]}" +" "
        if turn[1] == "":
            prefix += f"Assistant:" 
            return prefix
        else:
            prefix += f"Assistant: {turn[1]}" +" "

    return prefix



mapper = {
          "persona": {"shot_converter":convert_sample_to_shot_selector, 
                     "file_data":"../data/persona/","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:5},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
          "msc": {"shot_converter":convert_sample_to_shot_selector, 
                     "file_data":"../data/msc/session-2-","with_knowledge":None,
                     "shots":{1024:[0,1],2048:[0,1,3]},"max_shot":{1024:1,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":3},
          "wow": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"../data/wow/","with_knowledge":True,
                  "shots":{1024:[0,1,2],2048:[4,3,2,1,0]},"max_shot":{1024:1,2048:1},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":60,"max_number_turns":5},
          "wit": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"../data/wit/","with_knowledge":True,
                  "shots":{1024:[0,1],2048:[0,1,2,3]},"max_shot":{1024:1,2048:3},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":60,"max_number_turns":4},
          "ed": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"../data/ed/","with_knowledge":None,
                  "shots":{1024:[0,1,7],2048:[0,1,17]},"max_shot":{1024:7,2048:17},
                  "shot_separator":"\n\n",
                  "meta_type":"none","gen_len":50,"max_number_turns":5},
          "dialKG": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"../data/dialKG/","with_knowledge":True,
                  "shots":{1024:[0,1,3],2048:[0,1,9]},"max_shot":{1024:3,2048:9},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":4},
          "DD": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"../data/dailydialog/","with_knowledge":False,
                  "shots":{1024:[0,1,2],2048:[0,1,6]},"max_shot":{1024:2,2048:6},
                  "shot_separator":"\n\n",
                  "meta_type":"all_turns","gen_len":50,"max_number_turns":5},
        "smd-navigate": {"shot_converter":convert_sample_to_shot_selector, 
                     "file_data":"../data/smd/navigate-","with_knowledge":None,
                     "shots":{1024:[0,1],2048:[0,1,8]},"max_shot":{1024:1,2048:8},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
        "smd-schedule": {"shot_converter":convert_sample_to_shot_selector, 
                     "file_data":"../data/smd/schedule-","with_knowledge":None,
                     "shots":{1024:[0,1],2048:[0,1,8]},"max_shot":{1024:1,2048:8},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
        "smd-weather": {"shot_converter":convert_sample_to_shot_selector, 
                     "file_data":"../data/smd/weather-","with_knowledge":None,
                     "shots":{1024:[0,1],2048:[0,1,8]},"max_shot":{1024:1,2048:8},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
          "IC": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"../data/image_chat/","with_knowledge":False,
                  "shots":{1024:[0,1,5],2048:[0,1,10]},"max_shot":{1024:5,2048:10},
                  "shot_separator":"\n\n",
                  "meta_type":"all_turns_category","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-hotel": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"../data/mwoz/hotel-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-taxi": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"../data/mwoz/taxi-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-train": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"../data/mwoz/train-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-attraction": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"../data/mwoz/attraction-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-restaurant": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"../data/mwoz/restaurant-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
         }





def load_prefix(shots_value, shot_converter, 
                file_shot, name_dataset,sample_times=5):
    prefix_list = []
    for i in range(sample_times):
        shots = 0
        prefix_shot = {s:"" for s in shots_value}
        data = json.load(open(file_shot,"r"))
        random.Random(i).shuffle(data)
        prefix = []
        for d in data:
            prefix.append(d['dialogue'])
            shots += 1
            if shots in prefix_shot:
                prefix_shot[shots] = copy.copy(prefix)
        print(f"Loaded {name_dataset} {prefix_shot.keys()} shots for shuffle {i}!")
        prefix_list.append(prefix_shot)
    return prefix_list

def get_dataset_training(repetition):
    shot_list = [7,8,9,10]
    # shot_list = [1,2,3,4,5,6,10,100,500]
    available_datasets = list(mapper.keys())
    print(available_datasets)
    prefix_dict = {}
    for d in available_datasets:
        if "smd" in d:
            prefix_dict[d] = load_prefix(shots_value=shot_list, 
                        shot_converter=mapper[d]["shot_converter"], 
                        file_shot=mapper[d]["file_data"]+"train.json", 
                        name_dataset=d,sample_times=3)
        else:
            prefix_dict[d] = load_prefix(shots_value=shot_list, 
                        shot_converter=mapper[d]["shot_converter"], 
                        file_shot=mapper[d]["file_data"]+"valid.json", 
                        name_dataset=d,sample_times=3)

    id_g = 0
    data = {shot:[] for shot in shot_list}

    for dname, dial_shot in prefix_dict.items():
        print(f"Processing {dname}")
        for shots_k in shot_list:
            temp_dial = {"dialogue":[]}          
            for dial in dial_shot[repetition][shots_k]:
                for turn in dial:
                    data_dict = {"sentence":"", "label": ""}
                    temp_dial["dialogue"].append([turn[0],""])
                    data_dict['sentence'] = convert_sample_to_shot_selector(temp_dial)
                    data_dict['label'] = available_datasets.index(dname)
                    data[shots_k].append(data_dict)
                    temp_dial["dialogue"][-1][1] = turn[1]
                    id_g += 1

    # dataset_dict = {}
    # for shots_k, data_ in data_dict.items():
    #     dataset_dict[shots_k] = Dataset.from_dict(data_)
    return data


def get_dataset_train_all():
    
    available_datasets = list(mapper.keys())
    print(available_datasets)
    prefix_dict = defaultdict(list)
    for d in available_datasets:
        if "smd" in d:
            for dial in json.load(open(mapper[d]["file_data"]+"train.json","r")):
                prefix_dict[d].append(dial['dialogue'])
        else:
            for dial in json.load(open(mapper[d]["file_data"]+"valid.json","r")):
                prefix_dict[d].append(dial['dialogue'])

    id_g = 0
    data = [{"id":[],"sentence":[], "label": []}]

    for dname, dialogue in prefix_dict.items():
        temp_dial = {"dialogue":[]}          
        # random.Random(1234).shuffle(dialogue)
        for id_dial, dial in enumerate(dialogue):
            for turn in dial:
                data_dict = {"sentence":"", "label": ""}
                temp_dial["dialogue"].append([turn[0],""])
                data_dict['sentence'] = convert_sample_to_shot_selector(temp_dial)
                data_dict['label'] = available_datasets.index(dname)
                temp_dial["dialogue"][-1][1] = turn[1]
                data.append(data_dict)
                id_g += 1
            # if id_dial == 10: break
    # return Dataset.from_dict(data_dict)
    return data

def get_dataset_valid():
    
    available_datasets = list(mapper.keys())
    print(available_datasets)
    prefix_dict = defaultdict(list)
    for d in available_datasets:
        if "smd" in d:
            for dial in json.load(open(mapper[d]["file_data"]+"train.json","r")):
                prefix_dict[d].append(dial['dialogue'])
        else:
            for dial in json.load(open(mapper[d]["file_data"]+"valid.json","r")):
                prefix_dict[d].append(dial['dialogue'])


    id_g = 0
    data = []
    for dname, dialogue in prefix_dict.items():
        temp_dial = {"dialogue":[]}          
        random.Random(1234).shuffle(dialogue)
        for id_dial, dial in enumerate(dialogue):
            for turn in dial:
                data_dict = {"sentence":"", "label": ""}
                temp_dial["dialogue"].append([turn[0],""])
                # data_dict['id'] = id_g
                data_dict['sentence'] = convert_sample_to_shot_selector(temp_dial)
                data_dict['label'] = available_datasets.index(dname)
                data.append(data_dict)
                temp_dial["dialogue"][-1][1] = turn[1]
                id_g += 1
            if id_dial == 10: break
    # return Dataset.from_dict(data_dict)
    return data


def get_dataset_test():
    
    available_datasets = list(mapper.keys())
    print(available_datasets)
    prefix_dict = defaultdict(list)
    for d in available_datasets:
        for dial in json.load(open(mapper[d]["file_data"]+"test.json","r")):
            prefix_dict[d].append(dial['dialogue'])


    id_g = 0
    data = []
    for dname, dialogue in prefix_dict.items():
        temp_dial = {"dialogue":[]}          
        for id_dial, dial in enumerate(dialogue):
            for turn in dial:
                data_dict = {"sentence":"", "label": ""}
                temp_dial["dialogue"].append([turn[0],""])
                data_dict['sentence'] = convert_sample_to_shot_selector(temp_dial)
                data_dict['label'] = available_datasets.index(dname)
                temp_dial["dialogue"][-1][1] = turn[1]
                data.append(data_dict)
                id_g += 1
            if id_dial == 101: break
    return data

def saving(filename,file_data):
    with jsonlines.open(f'data/{filename}.json', mode='w') as writer:
        for d in file_data:
            writer.write(d)


if __name__ == "__main__":    
    for rep in [0,1,2]:
        dataset_train = get_dataset_training(repetition=rep)
        for shot, dataset_shot in dataset_train.items():
            saving(f"train_{shot}_{rep}",dataset_shot)

    dataset_valid = get_dataset_valid()
    dataset_test = get_dataset_test()
    saving("valid",dataset_valid)
    saving("test",dataset_test)

    # TODO run all training set ==> usually the performance are not that good!!
    # saving("train",dataset_train_all)
    # dataset_train_all = get_dataset_train_all()
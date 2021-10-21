import json
import os
import argparse
import random
import numpy as np
from utils.utils import load_model, save_file, checker_file
from metric.general import metric_report, argmin
from prompts.generic_prompt import load_prefix, evalute_prompt_prob
from prompts.skill_selector import convert_sample_to_shot_selector
from tabulate import tabulate
from collections import defaultdict
import os
from tqdm import tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)

mapper = {
          "persona": {"shot_converter":convert_sample_to_shot_selector, 
                     "file_data":"data/persona/","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:5},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
          "msc": {"shot_converter":convert_sample_to_shot_selector, 
                     "file_data":"data/msc/session-2-","with_knowledge":None,
                     "shots":{1024:[0,1],2048:[0,1,3]},"max_shot":{1024:1,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":3},
          "wow": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"data/wow/","with_knowledge":True,
                  "shots":{1024:[0,1,2],2048:[4,3,2,1,0]},"max_shot":{1024:1,2048:1},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":60,"max_number_turns":5},
          "wit": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"data/wit/","with_knowledge":True,
                  "shots":{1024:[0,1],2048:[0,1,2,3]},"max_shot":{1024:1,2048:3},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":60,"max_number_turns":4},
          "ed": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"data/ed/","with_knowledge":None,
                  "shots":{1024:[0,1,7],2048:[0,1,17]},"max_shot":{1024:7,2048:17},
                  "shot_separator":"\n\n",
                  "meta_type":"none","gen_len":50,"max_number_turns":5},
          "dialKG": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"data/dialKG/","with_knowledge":True,
                  "shots":{1024:[0,1,3],2048:[0,1,9]},"max_shot":{1024:3,2048:9},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":4},
          "DD": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"data/dailydialog/","with_knowledge":False,
                  "shots":{1024:[0,1,2],2048:[0,1,6]},"max_shot":{1024:2,2048:6},
                  "shot_separator":"\n\n",
                  "meta_type":"all_turns","gen_len":50,"max_number_turns":5},
        "smd-navigate": {"shot_converter":convert_sample_to_shot_selector, 
                     "file_data":"data/smd/navigate-","with_knowledge":None,
                     "shots":{1024:[0,1],2048:[0,1,8]},"max_shot":{1024:1,2048:8},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
        "smd-schedule": {"shot_converter":convert_sample_to_shot_selector, 
                     "file_data":"data/smd/schedule-","with_knowledge":None,
                     "shots":{1024:[0,1],2048:[0,1,8]},"max_shot":{1024:1,2048:8},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
        "smd-weather": {"shot_converter":convert_sample_to_shot_selector, 
                     "file_data":"data/smd/weather-","with_knowledge":None,
                     "shots":{1024:[0,1],2048:[0,1,8]},"max_shot":{1024:1,2048:8},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
          "IC": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"data/image_chat/","with_knowledge":False,
                  "shots":{1024:[0,1,5],2048:[0,1,10]},"max_shot":{1024:5,2048:10},
                  "shot_separator":"\n\n",
                  "meta_type":"all_turns_category","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-hotel": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"data/mwoz/hotel-","level":"dialogue","with_knowledge":False,
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-taxi": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"data/mwoz/taxi-","level":"dialogue","with_knowledge":False,
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-train": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"data/mwoz/train-","level":"dialogue","with_knowledge":False,
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-attraction": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"data/mwoz/attraction-","level":"dialogue","with_knowledge":False,
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-restaurant": {"shot_converter":convert_sample_to_shot_selector, 
                 "file_data":"data/mwoz/restaurant-","level":"dialogue","with_knowledge":False,
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
         }


## This is the config dictionary used to select the template converter
mapper_safety = {
          "unsa_topic": {"file_data":"data/safety_layers/safety_topic.json","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":2},
          "unsa_nonadv": {"file_data":"data/safety_layers/safety_nonadv.json","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":2},
          "unsa_adv": {"file_data":"data/safety_layers/safety_adv.json","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":2},
         }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", default="gpt2",type=str,required=True)
    parser.add_argument("--dataset", default="persona",type=str,required=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--sample_times", type=int, default=3)
    parser.add_argument("--shots_k", type=int, default=1)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--do_sample", action='store_true', help="sample n times and rescore based on ppl")
    parser.add_argument("--multigpu", action='store_true', help="run on multiple gpus")
    parser.add_argument("--verbose", action='store_true', help="run on multiple gpus")
    parser.add_argument("--safety", action='store_true', help="run on multiple gpus")

    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    beam = args.beam
    model_checkpoint = args.model_checkpoint

    model, tokenizer, max_seq = load_model(args,model_checkpoint,device)
    
    available_datasets = mapper.keys()
    number_of_classes = len(available_datasets)
    print(available_datasets)
    prefix_dict = {}
    for d in available_datasets:
        prefix_dict[d] = load_prefix(tokenizer=tokenizer, shots_value=[args.shots_k], 
                    shot_converter=mapper[d]["shot_converter"], 
                    file_shot= mapper[d]["file_data"]+"train.json" if "smd" in d else mapper[d]["file_data"]+"valid.json", 
                    name_dataset=d, with_knowledge=mapper[d]["with_knowledge"], 
                    shot_separator=mapper[d]["shot_separator"],sample_times=args.sample_times)

    if args.safety:
       ## add safety prompts
       for d in mapper_safety.keys():
              prefix_dict[d] = load_prefix(tokenizer=tokenizer, shots_value=[args.shots_k], 
                     shot_converter=convert_sample_to_shot_selector, 
                     file_shot= mapper_safety[d]["file_data"], 
                     name_dataset=d, with_knowledge=None, 
                     shot_separator=mapper_safety[d]["shot_separator"],sample_times=args.sample_times)

    for shots_k in [args.shots_k]:
        if checker_file(f"{model_checkpoint}_{shots_k}_{args.repetition}.json"):
            y_test = []
            y_pred = []
            for i_d, d in enumerate(available_datasets):
                if d not in mapper_safety.keys():
                    results_to_score = evalute_prompt_prob(model, tokenizer, shot_converter=mapper[d]["shot_converter"], 
                                        file_to_eval=mapper[d]["file_data"]+"test.json", 
                                        prefix=prefix_dict, device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                        with_knowledge=mapper[d]["with_knowledge"], max_seq=max_seq,
                                        max_shot=shots_k,
                                        meta_type=mapper[d]["meta_type"], verbose=args.verbose, repetition=args.repetition)
                    for res in results_to_score:
                        pred_id = argmin(list(dict(res).values()))
                        y_test.append(i_d)
                        y_pred.append(pred_id)
                    
            # print(f"SHOT: {shots_k}")
            score = metric_report(y_test, y_pred)
            save_file(f"{model_checkpoint}_{shots_k}_{args.repetition}.json", {"score":score,"prediction":{"y_test":y_test,"y_pred":y_pred}})

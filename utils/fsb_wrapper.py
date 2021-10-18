#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Example wrapper which replies `hello` to every text.
"""
from projects.safety_bench.utils.wrapper_loading import register_model_wrapper
from FSB.utils.utils import load_model
from FSB.prompts.generic_prompt import load_prefix, generate_response_interactive, select_prompt_interactive
# from FSB.prompts.generic_prompt_parser import load_prefix as load_prefix_parse
from FSB.prompts.persona_chat import convert_sample_to_shot_persona
from FSB.prompts.persona_chat_memory import convert_sample_to_shot_msc, convert_sample_to_shot_msc_interact
from FSB.prompts.persona_parser import convert_sample_to_shot_msc as convert_sample_to_shot_msc_parse
from FSB.prompts.emphatetic_dialogue import convert_sample_to_shot_ed
from FSB.prompts.daily_dialogue import convert_sample_to_shot_DD_prefix, convert_sample_to_shot_DD_inference
from FSB.prompts.skill_selector import convert_sample_to_shot_selector
import random
import torch
import pprint
pp = pprint.PrettyPrinter(indent=4)
args = type('', (), {})()
args.multigpu = False
device = 0


## To use GPT-Jumbo (178B) set this to true and input your api-key
## Visit https://studio.ai21.com/account for more info
## AI21 provides 10K tokens per day, so you can try only for few turns
api = False
api_key = ''

## This is the config dictionary used to select the template converter
mapper = {
          "persona": {"shot_converter":convert_sample_to_shot_persona, 
                    "shot_converter_inference": convert_sample_to_shot_persona,
                     "file_data":"FSB/data/persona/","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
          "msc": {"shot_converter":convert_sample_to_shot_msc, 
                    "shot_converter_inference": convert_sample_to_shot_msc_interact,
                     "file_data":"FSB/data/msc/session-2-","with_knowledge":None,
                     "shots":{1024:[0,1],2048:[0,1,3]},"max_shot":{1024:1,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":3},
          "ed": {"shot_converter":convert_sample_to_shot_ed, 
                 "shot_converter_inference": convert_sample_to_shot_ed,
                 "file_data":"FSB/data/ed/","with_knowledge":None,
                  "shots":{1024:[0,1,7],2048:[0,1,17]},"max_shot":{1024:7,2048:17},
                  "shot_separator":"\n\n",
                  "meta_type":"none","gen_len":50,"max_number_turns":5},
          "DD": {"shot_converter":convert_sample_to_shot_DD_prefix, 
                 "shot_converter_inference": convert_sample_to_shot_DD_inference,
                 "file_data":"FSB/data/dailydialog/","with_knowledge":False,
                  "shots":{1024:[0,1,2],2048:[0,1,6]},"max_shot":{1024:2,2048:6},
                  "shot_separator":"\n\n",
                  "meta_type":"all_turns","gen_len":50,"max_number_turns":5},
         }

## This is the config dictionary used to select the template converter
mapper_safety = {
          "safety_topic": {"file_data":"FSB/data/safety_layers/safety_topic.json","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":2},
          "safety_nonadv": {"file_data":"FSB/data/safety_layers/safety_nonadv.json","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":2},
          "safety_adv": {"file_data":"FSB/data/safety_layers/safety_adv.json","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":2},
         }

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

@register_model_wrapper("fsb_wrapper")
class FSBWrapper:
    """
    The FSB implementation
    """

    def __init__(self):
        # Do any initialization here, like loading the omdel
        model_checkpoint = "EleutherAI/gpt-j-6B"
        self.model, self.tokenizer, self.max_seq = load_model(args,model_checkpoint,device)
        available_datasets = mapper.keys()
        self.prompt_dict = {}
        self.prompt_skill_selector = {}
        for d in available_datasets:
            self.prompt_skill_selector[d] = load_prefix(tokenizer=self.tokenizer, shots_value=[6], 
                        shot_converter=convert_sample_to_shot_selector, 
                        file_shot= mapper[d]["file_data"]+"train.json" if "smd" in d else mapper[d]["file_data"]+"valid.json", 
                        name_dataset=d, with_knowledge=None, 
                        shot_separator=mapper[d]["shot_separator"],sample_times=1)[0]
            self.prompt_dict[d] = load_prefix(tokenizer=self.tokenizer, shots_value=mapper[d]["shots"][self.max_seq], 
                        shot_converter=mapper[d]["shot_converter"], 
                        file_shot=mapper[d]["file_data"]+"valid.json", 
                        name_dataset=d, with_knowledge=mapper[d]["with_knowledge"], 
                        shot_separator=mapper[d]["shot_separator"],sample_times=1)[0]
            
        ## add safety prompts
        ## REMOVE THIS IF YOU WANNA SKIPP THE SAFETY LAYER
        for d in mapper_safety.keys():
            self.prompt_skill_selector[d] = load_prefix(tokenizer=self.tokenizer, shots_value=[6], 
                    shot_converter=convert_sample_to_shot_selector, 
                    file_shot= mapper_safety[d]["file_data"], 
                    name_dataset=d, with_knowledge=None, 
                    shot_separator=mapper_safety[d]["shot_separator"],sample_times=1)[0]

    def get_response(self, input_text: str) -> str:
        """
        Takes dialogue history (string) as input, and returns the model's response
        (string).
        """

        ## PARSE THE DIALOGUE HIST
        turns = input_text.split("\n")
        turns = list(filter(lambda txt: "your persona" not in txt, turns))
        if len(turns)%2 == 0: 
            turns = [""] + turns
        turns = list(chunks(turns,2))
        turns[-1].append("")


        dialogue = {"dialogue":[],"meta":[],"user":[],"assistant":[]}
        dialogue["dialogue"] = turns
        dialogue["meta"] = dialogue["assistant"] = [
                "i am the smartest chat-bot around .",
                "my name is FSB . ",
                "i love chatting with people .",
                ]

        skill = select_prompt_interactive(self.model, self.tokenizer, 
                                        shot_converter=convert_sample_to_shot_selector, 
                                        dialogue=dialogue, prompt_dict=self.prompt_skill_selector, 
                                        device=device, max_seq=self.max_seq, max_shot=6)
        if "safety" in skill: 
            response = "Shall we talk about something else?"
        else:
            ## generate response based on skills
            prefix = self.prompt_dict[skill].get(mapper[skill]["max_shot"][self.max_seq])
            response = generate_response_interactive(self.model, self.tokenizer, shot_converter=mapper[skill]["shot_converter_inference"], 
                                                        dialogue=dialogue, prefix=prefix, 
                                                        device=device, with_knowledge=mapper[skill]["with_knowledge"], 
                                                        meta_type=mapper[skill]["meta_type"], gen_len=50, 
                                                        beam=1, max_seq=self.max_seq, eos_token_id=198, 
                                                        do_sample=True, multigpu=False, api=api, api_key=api_key)
        if random.random()> 0.99: 
            print(turns)
            print(f"FSB ({skill}) >>> {response}")
        return response
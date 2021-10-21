#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from projects.safety_bench.utils.wrapper_loading import register_model_wrapper
from FSB.utils.utils import load_model
from FSB.prompts.generic_prompt import load_prefix, generate_response_interactive, select_prompt_interactive
from FSB.prompts.generic_prompt_parser import load_prefix as load_prefix_parse
from FSB.prompts.generic_prompt_parser import generate_response_DKG_interactive
from FSB.prompts.persona_chat import convert_sample_to_shot_persona
from FSB.prompts.persona_chat_memory import convert_sample_to_shot_msc, convert_sample_to_shot_msc_interact
from FSB.prompts.emphatetic_dialogue import convert_sample_to_shot_ed
from FSB.prompts.daily_dialogue import convert_sample_to_shot_DD_prefix, convert_sample_to_shot_DD_inference
from FSB.prompts.persona_parser import convert_sample_to_shot_msc as convert_sample_to_shot_msc_parse
from FSB.prompts.wizard_of_wikipedia import convert_sample_to_shot_wow, convert_sample_to_shot_wow_interact
from FSB.prompts.wizard_of_wikipedia_parse import convert_sample_to_shot_wow as convert_sample_to_shot_wow_parse
from FSB.prompts.wizard_of_internet import convert_sample_to_shot_wit, convert_sample_to_shot_wit_interact
from FSB.prompts.wizard_of_internet_parser import convert_sample_to_shot_wit as convert_sample_to_shot_wit_parse
from FSB.prompts.dialKG import convert_sample_to_shot_dialKG, convert_sample_to_shot_dialKG_interact
from FSB.prompts.dialKG_parser import convert_sample_to_shot_dialKG as convert_sample_to_shot_dialKG_parse
from FSB.prompts.skill_selector import convert_sample_to_shot_selector
from FSB.utils.wit_parlai_retriever import SearchEngineRetriever
from py2neo import Graph
import wikipedia
import random
import torch
import pprint
from nltk.tokenize import sent_tokenize
pp = pprint.PrettyPrinter(indent=4)
args = type('', (), {})()
args.multigpu = False
device = 0
ks = SearchEngineRetriever(search_server="http://eez114.ece.ust.hk:8081")
kg = Graph("http://eez114.ece.ust.hk:7474", auth=("neo4j", "CAiRE2020neo4j"))

## To use GPT-Jumbo (178B) set this to true and input your api-key
## Visit https://studio.ai21.com/account for more info
## AI21 provides 10K tokens per day, so you can try only for few turns
api = False
api_key = ''

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
          "safe": {"shot_converter":convert_sample_to_shot_persona, 
                 "shot_converter_inference": convert_sample_to_shot_persona,
                 "file_data":"FSB/data/safety_layers/safety_safe_adv_","with_knowledge":None,
                  "shots":{1024:[0,1,5],2048:[0,1,10]},"max_shot":{1024:5,2048:10},
                  "shot_separator":"\n\n",
                  "meta_type":"none","gen_len":50,"max_number_turns":5},
          "DD": {"shot_converter":convert_sample_to_shot_DD_prefix, 
                 "shot_converter_inference": convert_sample_to_shot_DD_inference,
                 "file_data":"FSB/data/dailydialog/","with_knowledge":False,
                  "shots":{1024:[0,1,2],2048:[0,1,6]},"max_shot":{1024:2,2048:6},
                  "shot_separator":"\n\n",
                  "meta_type":"all_turns","gen_len":50,"max_number_turns":5},
          "wow": {"shot_converter":convert_sample_to_shot_wow, 
                 "shot_converter_inference": convert_sample_to_shot_wow_interact,
                 "file_data":"FSB/data/wow/","with_knowledge":True,
                  "shots":{1024:[0,1,2],2048:[4,3,2,1,0]},"max_shot":{1024:1,2048:3},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":60,"max_number_turns":5},
          "wit": {"shot_converter":convert_sample_to_shot_wit, 
                 "shot_converter_inference": convert_sample_to_shot_wit_interact,
                 "file_data":"FSB/data/wit/","with_knowledge":True,
                  "shots":{1024:[0,1],2048:[0,1,2,3]},"max_shot":{1024:1,2048:3},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":60,"max_number_turns":4},
          "dialKG": {"shot_converter":convert_sample_to_shot_dialKG, 
                 "shot_converter_inference": convert_sample_to_shot_dialKG_interact,
                 "file_data":"FSB/data/dialKG/","with_knowledge":True,
                  "shots":{1024:[0,1,3],2048:[0,1,9]},"max_shot":{1024:3,2048:9},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":4},
          "wow-parse": {"shot_converter":convert_sample_to_shot_wow_parse, 
                 "file_data":"FSB/data/wow/parse-","level":"dialogue", "retriever":"wiki",
                  "shots":{1024:[0, 1, 5],2048:[0, 1, 5, 10]},"max_shot":{1024:5,2048:10},
                  "shot_separator":"\n\n", "meta_type":"last_turn","gen_len":50,"max_number_turns":2},
          "wit-parse": {"shot_converter":convert_sample_to_shot_wit_parse, 
                 "file_data":"FSB/data/wit/","level":"dialogue","max_shot":{1024:1,2048:4},
                  "shots":{1024:[0,1],2048:[0, 1, 2, 3, 4]},"shot_separator":"\n\n", "retriever":"internet",
                  "meta_type":"query","gen_len":50,"max_number_turns":2},
          "dialKG-parse": {"shot_converter":convert_sample_to_shot_dialKG_parse, 
                 "file_data":"FSB/data/dialKG/","level":"dialogue", "max_shot":{1024:3,2048:9},
                  "shots":{1024:[0,1,2,3],2048:[0, 1, 5, 9]},"shot_separator":"\n\n", "retriever":"graph",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":5},
          "msc-parse": {"shot_converter":convert_sample_to_shot_msc_parse, "max_shot":{1024:1,2048:2},
                 "file_data":"FSB/data/msc/parse-session-1-","level":"dialogue", "retriever":"none",
                  "shots":{1024:[0,1],2048:[0, 1, 2]},"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":3},
               
         }
## This is the config dictionary used to select the template converter
mapper_safety = {
          "unsa_topic": {"file_data":"FSB/data/safety_layers/safety_topic.json","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":2},
          "unsa_nonadv": {"file_data":"FSB/data/safety_layers/safety_nonadv.json","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":2},
          "unsa_adv": {"file_data":"FSB/data/safety_layers/safety_adv.json","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":2},
         }

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

@register_model_wrapper("fsb_wrapper_full")
class FSBWrapper:
    """
    The FSB implementation
    """

    def __init__(self):
        # Do any initialization here, like loading the omdel
        model_checkpoint = "EleutherAI/gpt-j-6B"
        # model_checkpoint = "gpt2"
        self.model, self.tokenizer, self.max_seq = load_model(args,model_checkpoint,device)
        available_datasets = mapper.keys()
        self.prompt_dict = {}
        self.prompt_parse = {}
        self.prompt_skill_selector = {}
        for d in available_datasets:
            if "parse" in d:
                self.prompt_parse[d] = load_prefix_parse(tokenizer=self.tokenizer, shots_value=mapper[d]["shots"][self.max_seq], 
                                        shot_converter=mapper[d]["shot_converter"], 
                                        file_shot=mapper[d]["file_data"]+"valid.json", 
                                        name_dataset=d, level=mapper[d]["level"], 
                                        shot_separator=mapper[d]["shot_separator"],sample_times=1)[0]
            else:
                if "safe" != d:
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


    def run_parsers(self, args, model, tokenizer, device, max_seq, dialogue, skill, prefix_dict):

        if skill not in ["msc", "wow", "wit","dialKG"]: return dialogue

        ### parse 
        d_p = f"{skill}-parse"
        print(f"Parse with {d_p}")

        prefix = prefix_dict[d_p].get(mapper[d_p]["max_shot"][max_seq])
        if skill == "dialKG":
            ### THIS REQUIRE A NEO4J DB UP and RUNNING
            query = generate_response_DKG_interactive(model, tokenizer, shot_converter=mapper[d_p]["shot_converter"], 
                                        dialogue=dialogue, prefix=prefix, 
                                        device=device,
                                        level=mapper[d_p]["level"], gen_len=50, 
                                        beam=1, max_seq=max_seq, eos_token_id=198, 
                                        do_sample=False, multigpu=False, 
                                        verbose=False, KG=kg)
        else:
            query = generate_response_interactive(model, tokenizer, shot_converter=mapper[d_p]["shot_converter"], 
                                                        dialogue=dialogue, prefix=prefix, 
                                                        device=device,  with_knowledge=None, 
                                                        meta_type=None, gen_len=50, 
                                                        beam=1, max_seq=max_seq, eos_token_id=198, 
                                                        do_sample=False, multigpu=False, api=api, api_key=api_key)

        print(f"Query: {query}")
        if query.lower() == "none": return dialogue

        if skill == "wow" and query not in dialogue["query_mem"]:
            dialogue["query_mem"].append(query)
            ## Try first with Wiki
            try:
                retrieve_K = wikipedia.summary(query, sentences=1)
            except:
                ## Then try with the Internet
                try: 
                    page = ks.retrieve(queries=[query], num_ret=1)[0]
                    retrieve_K = sent_tokenize(page[0]['content'])[1] 
                except:
                    retrieve_K = "None"
            dialogue["KB_wiki"][-1] = [retrieve_K]
        elif skill == "wit" and query not in dialogue["query_mem"]:
            dialogue["query_mem"].append(query)
            try: 
                page = ks.retrieve(queries=[query], num_ret=1)[0]
                retrieve_K = sent_tokenize(page[0]['content'])[1] 
            except:
                retrieve_K = "None"
            dialogue["KB_internet"][-1] = [retrieve_K]
            dialogue["query"][-1] = [query]
        elif skill == "dialKG":
            dialogue["KG"][-1] = [query]
        elif skill == "msc":
            dialogue["user"].append(query)
            dialogue["user_memory"][-1] = [query]
        return dialogue

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


        dialogue = {"dialogue":[],"meta":[],"user":[],
                    "assistant":[],"user_memory":[], 
                    "KG":[], "KB_internet": [], 
                    "KB_wiki": [], "query":[],
                    "query_mem":[]}
        dialogue["dialogue"] = turns
        dialogue["meta"] = dialogue["assistant"] = [
                "i am the smartest chat-bot around .",
                "my name is FSB . ",
                "i love chatting with people .",
                ]

        for _ in range(len(turns)):        
            dialogue["user_memory"].append([])
            dialogue["KB_wiki"].append([])
            dialogue["KB_internet"].append([])
            dialogue["query"].append([])
            dialogue["KG"].append([])

        skill = select_prompt_interactive(self.model, self.tokenizer, 
                                    shot_converter=convert_sample_to_shot_selector, 
                                    dialogue=dialogue, prompt_dict=self.prompt_skill_selector, 
                                    device=device, max_seq=self.max_seq, max_shot=6, sample=False)
        

        if "unsa" in skill: 
            skill = "safe"
            ## generate response based on skills
            prefix = self.prompt_dict[skill].get(mapper[skill]["max_shot"][self.max_seq])
            response = generate_response_interactive(self.model, self.tokenizer, shot_converter=mapper[skill]["shot_converter_inference"], 
                                                        dialogue=dialogue, prefix=prefix, 
                                                        device=device, with_knowledge=mapper[skill]["with_knowledge"], 
                                                        meta_type=mapper[skill]["meta_type"], gen_len=50, 
                                                        beam=1, max_seq=self.max_seq, eos_token_id=198, 
                                                        do_sample=True, multigpu=False, api=api, api_key=api_key)
        else:
            dialogue = self.run_parsers(args, self.model, self.tokenizer, device=device, max_seq=self.max_seq,
                                dialogue=dialogue, skill=skill,  
                                prefix_dict=self.prompt_parse)
            ## generate response based on skills
            prefix = self.prompt_dict[skill].get(mapper[skill]["max_shot"][self.max_seq])
            response = generate_response_interactive(self.model, self.tokenizer, shot_converter=mapper[skill]["shot_converter_inference"], 
                                                        dialogue=dialogue, prefix=prefix, 
                                                        device=device, with_knowledge=mapper[skill]["with_knowledge"], 
                                                        meta_type=mapper[skill]["meta_type"], gen_len=50, 
                                                        beam=1, max_seq=self.max_seq, eos_token_id=198, 
                                                        do_sample=True, multigpu=False, api=api, api_key=api_key)
        if random.random()> 0.9: 
            print(turns)
            print(f"FSB ({skill}) >>> {response}")
        return response
from utils.utils import load_model
from prompts.generic_prompt import load_prefix, generate_response_interactive, select_prompt_interactive
from prompts.generic_prompt_parser import load_prefix as load_prefix_parse
from prompts.generic_prompt_parser import generate_response_DKG_interactive
from prompts.persona_chat import convert_sample_to_shot_persona
from prompts.persona_chat_memory import convert_sample_to_shot_msc, convert_sample_to_shot_msc_interact
from prompts.emphatetic_dialogue import convert_sample_to_shot_ed
from prompts.daily_dialogue import convert_sample_to_shot_DD_prefix, convert_sample_to_shot_DD_inference
from prompts.persona_parser import convert_sample_to_shot_msc as convert_sample_to_shot_msc_parse
from prompts.wizard_of_wikipedia import convert_sample_to_shot_wow, convert_sample_to_shot_wow_interact
from prompts.wizard_of_wikipedia_parse import convert_sample_to_shot_wow as convert_sample_to_shot_wow_parse
from prompts.wizard_of_internet import convert_sample_to_shot_wit, convert_sample_to_shot_wit_interact
from prompts.wizard_of_internet_parser import convert_sample_to_shot_wit as convert_sample_to_shot_wit_parse
from prompts.dialKG import convert_sample_to_shot_dialKG, convert_sample_to_shot_dialKG_interact
from prompts.dialKG_parser import convert_sample_to_shot_dialKG as convert_sample_to_shot_dialKG_parse
from prompts.skill_selector import convert_sample_to_shot_selector
from utils.wit_parlai_retriever import SearchEngineRetriever
from py2neo import Graph
import wikipedia
import random
import torch
import pprint
from nltk.tokenize import sent_tokenize


pp = pprint.PrettyPrinter(indent=4)
args = type('', (), {})()
args.multigpu = False
device = 1
safety_level = 6
shot_selector = 6
sample_skill = False

## Check the retriever forlder for more info
ks = SearchEngineRetriever(search_server="ADDRESS")
kg = Graph("address", auth=("USR", "PWD"))

## To use GPT-Jumbo (178B) set this to true and input your api-key
## Visit https://studio.ai21.com/account for more info
## AI21 provides 10K tokens per day, so you can try only for few turns
api = False
api_key = ''


model_checkpoint = "EleutherAI/gpt-j-6B"
model, tokenizer, max_seq = load_model(args,model_checkpoint,device)


## This is the config dictionary used to select the template converter
## Remove dialKG if you don't have the Graph Neo4j
mapper = {
          "persona": {"shot_converter":convert_sample_to_shot_persona, 
                    "shot_converter_inference": convert_sample_to_shot_persona,
                     "file_data":"data/persona/","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
          "msc": {"shot_converter":convert_sample_to_shot_msc, 
                    "shot_converter_inference": convert_sample_to_shot_msc_interact,
                     "file_data":"data/msc/session-2-","with_knowledge":None,
                     "shots":{1024:[0,1],2048:[0,1,3]},"max_shot":{1024:1,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":3},
          "ed": {"shot_converter":convert_sample_to_shot_ed, 
                 "shot_converter_inference": convert_sample_to_shot_ed,
                 "file_data":"data/ed/","with_knowledge":None,
                  "shots":{1024:[0,1,7],2048:[0,1,17]},"max_shot":{1024:7,2048:17},
                  "shot_separator":"\n\n",
                  "meta_type":"none","gen_len":50,"max_number_turns":5},
          "safe": {"shot_converter":convert_sample_to_shot_persona, 
                 "shot_converter_inference": convert_sample_to_shot_persona,
                 "file_data":"data/safety_layers/safety_safe_adv_","with_knowledge":None,
                  "shots":{1024:[0,1,5],2048:[0,1,10]},"max_shot":{1024:5,2048:10},
                  "shot_separator":"\n\n",
                  "meta_type":"none","gen_len":50,"max_number_turns":5},
          "DD": {"shot_converter":convert_sample_to_shot_DD_prefix, 
                 "shot_converter_inference": convert_sample_to_shot_DD_inference,
                 "file_data":"data/dailydialog/","with_knowledge":False,
                  "shots":{1024:[0,1,2],2048:[0,1,6]},"max_shot":{1024:2,2048:6},
                  "shot_separator":"\n\n",
                  "meta_type":"all_turns","gen_len":50,"max_number_turns":5},
          "wow": {"shot_converter":convert_sample_to_shot_wow, 
                 "shot_converter_inference": convert_sample_to_shot_wow_interact,
                 "file_data":"data/wow/","with_knowledge":True,
                  "shots":{1024:[0,1,2],2048:[4,3,2,1,0]},"max_shot":{1024:1,2048:3},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":60,"max_number_turns":5},
          "wit": {"shot_converter":convert_sample_to_shot_wit, 
                 "shot_converter_inference": convert_sample_to_shot_wit_interact,
                 "file_data":"data/wit/","with_knowledge":True,
                  "shots":{1024:[0,1],2048:[0,1,2,3]},"max_shot":{1024:1,2048:3},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":60,"max_number_turns":4},
          "dialKG": {"shot_converter":convert_sample_to_shot_dialKG, 
                 "shot_converter_inference": convert_sample_to_shot_dialKG_interact,
                 "file_data":"data/dialKG/","with_knowledge":True,
                  "shots":{1024:[0,1,3],2048:[0,1,9]},"max_shot":{1024:3,2048:9},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":4},
          "wow-parse": {"shot_converter":convert_sample_to_shot_wow_parse, 
                 "file_data":"data/wow/parse-","level":"dialogue", "retriever":"wiki",
                  "shots":{1024:[0, 1, 5],2048:[0, 1, 5, 10]},"max_shot":{1024:5,2048:10},
                  "shot_separator":"\n\n", "meta_type":"last_turn","gen_len":50,"max_number_turns":2},
          "wit-parse": {"shot_converter":convert_sample_to_shot_wit_parse, 
                 "file_data":"data/wit/","level":"dialogue","max_shot":{1024:1,2048:4},
                  "shots":{1024:[0,1],2048:[0, 1, 2, 3, 4]},"shot_separator":"\n\n", "retriever":"internet",
                  "meta_type":"query","gen_len":50,"max_number_turns":2},
          "dialKG-parse": {"shot_converter":convert_sample_to_shot_dialKG_parse, 
                 "file_data":"data/dialKG/","level":"dialogue", "max_shot":{1024:3,2048:9},
                  "shots":{1024:[0,1,2,3],2048:[0, 1, 5, 9]},"shot_separator":"\n\n", "retriever":"graph",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":5},
          "msc-parse": {"shot_converter":convert_sample_to_shot_msc_parse, "max_shot":{1024:1,2048:2},
                 "file_data":"data/msc/parse-session-1-","level":"dialogue", "retriever":"none",
                  "shots":{1024:[0,1],2048:[0, 1, 2]},"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":3},
               
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
        ## THIS MAKE IT VERY VERY SAFE
        #   "unsa_adv": {"file_data":"data/safety_layers/safety_adv.json","with_knowledge":None,
        #              "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
        #              "shot_separator":"\n\n",
        #              "meta_type":"all","gen_len":50,"max_number_turns":2},
         }


### LOAD PROMPTS
available_datasets = mapper.keys()
prompt_dict = {}
prompt_parse = {}
prompt_skill_selector = {}
for d in available_datasets:
    if "parse" in d:
        prompt_parse[d] = load_prefix_parse(tokenizer=tokenizer, shots_value=mapper[d]["shots"][max_seq], 
                                shot_converter=mapper[d]["shot_converter"], 
                                file_shot=mapper[d]["file_data"]+"valid.json", 
                                name_dataset=d, level=mapper[d]["level"], 
                                shot_separator=mapper[d]["shot_separator"],sample_times=1)[0]
    else:
        if "safe" != d:
            prompt_skill_selector[d] = load_prefix(tokenizer=tokenizer, shots_value=[shot_selector], 
                        shot_converter=convert_sample_to_shot_selector, 
                        file_shot= mapper[d]["file_data"]+"train.json" if "smd" in d else mapper[d]["file_data"]+"valid.json", 
                        name_dataset=d, with_knowledge=None, 
                        shot_separator=mapper[d]["shot_separator"],sample_times=1)[0]
        prompt_dict[d] = load_prefix(tokenizer=tokenizer, shots_value=mapper[d]["shots"][max_seq], 
                    shot_converter=mapper[d]["shot_converter"], 
                    file_shot=mapper[d]["file_data"]+"valid.json", 
                    name_dataset=d, with_knowledge=mapper[d]["with_knowledge"], 
                    shot_separator=mapper[d]["shot_separator"],sample_times=1)[0]
    
## add safety prompts
for d in mapper_safety.keys():
    prompt_skill_selector[d] = load_prefix(tokenizer=tokenizer, shots_value=[safety_level], 
            shot_converter=convert_sample_to_shot_selector, 
            file_shot= mapper_safety[d]["file_data"], 
            name_dataset=d, with_knowledge=None, 
            shot_separator=mapper_safety[d]["shot_separator"],sample_times=1)[0]


def run_parsers(args, model, tokenizer, device, max_seq, dialogue, skill, prefix_dict):

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


max_number_turns = 5
dialogue = {"dialogue":[],"meta":[],"user":[],
            "assistant":[],"user_memory":[], 
            "KG":[], "KB_internet": [], 
            "KB_wiki": [], "query":[],
            "query_mem":[]}

## This meta information is the persona of the FSB
dialogue["meta"] = dialogue["assistant"] = [
                "i am the smartest chat-bot around .",
                "my name is FSB . ",
                "i love chatting with people .",
                "my creator is Andrea"
                ]
t = 10
while t>0: 
    t -= 1
    user_utt = input(">>> ")
    dialogue["dialogue"].append([user_utt,""])
    ## run the skill selector
    skill = select_prompt_interactive(model, tokenizer, 
                                    shot_converter=convert_sample_to_shot_selector, 
                                    dialogue=dialogue, prompt_dict=prompt_skill_selector, 
                                    device=device, max_seq=max_seq, max_shot=shot_selector, sample=sample_skill)
    dialogue["user_memory"].append([])
    dialogue["KB_wiki"].append([])
    dialogue["KB_internet"].append([])
    dialogue["query"].append([])
    dialogue["KG"].append([])

    if "unsa" in skill: 
        skill = "safe"
        ## generate response based on skills
        prefix = prompt_dict[skill].get(mapper[skill]["max_shot"][max_seq])
        response = generate_response_interactive(model, tokenizer, shot_converter=mapper[skill]["shot_converter_inference"], 
                                                    dialogue=dialogue, prefix=prefix, 
                                                    device=device, with_knowledge=mapper[skill]["with_knowledge"], 
                                                    meta_type=mapper[skill]["meta_type"], gen_len=50, 
                                                    beam=1, max_seq=max_seq, eos_token_id=198, 
                                                    do_sample=True, multigpu=False, api=api, api_key=api_key)
                    
    else:
        ## parse user dialogue history ==> msc-parse
        dialogue = run_parsers(args, model, tokenizer, device=device, max_seq=max_seq,
                                dialogue=dialogue, skill=skill,  
                                prefix_dict=prompt_parse)
        ## generate response based on skills
        prefix = prompt_dict[skill].get(mapper[skill]["max_shot"][max_seq])
        response = generate_response_interactive(model, tokenizer, shot_converter=mapper[skill]["shot_converter_inference"], 
                                                    dialogue=dialogue, prefix=prefix, 
                                                    device=device, with_knowledge=mapper[skill]["with_knowledge"], 
                                                    meta_type=mapper[skill]["meta_type"], gen_len=50, 
                                                    beam=1, max_seq=max_seq, eos_token_id=198, 
                                                    do_sample=True, multigpu=False, api=api, api_key=api_key)
                    

    print(f"FSB ({skill}) >>> {response}")
    dialogue["dialogue"][-1][1] = response
    dialogue["dialogue"] = dialogue["dialogue"][-max_number_turns:]
    dialogue["user_memory"] = dialogue["user_memory"][-max_number_turns:]
    dialogue["KB_wiki"] = dialogue["KB_wiki"][-max_number_turns:]
    dialogue["KB_internet"] = dialogue["KB_internet"][-max_number_turns:]
    dialogue["query"] = dialogue["query"][-max_number_turns:]
    dialogue["KG"] = dialogue["KG"][-max_number_turns:]
print("\n\nThis is the conversation history with its meta-data!\n\n")
print(pp.pprint(dialogue))
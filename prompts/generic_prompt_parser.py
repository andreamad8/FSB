import torch
import math
import numpy as np
import re
import json
from tqdm import tqdm
import logging
import copy
import random
import requests
from prompts.generic_prompt import get_response
from cleantext import clean
from nltk.tokenize import sent_tokenize
from utils.tfidf import TfIdf
from utils.wit_parlai_retriever import SearchEngineRetriever
from metric.general import normalize_answer
from py2neo import Graph, Node, Relationship
logging.getLogger('transformers.generation_utils').setLevel(logging.CRITICAL)


## we copied KILT inside our repo
from data.wow.KILT.kilt.knowledge_source import KnowledgeSource

## Put the neo4j information
ks = Graph("http://address:port", auth=("usrname", "pwd"))




def load_prefix(tokenizer, shots_value, shot_converter, 
                file_shot, name_dataset, level, 
                shot_separator="\n\n",sample_times=5):
    prefix_list = []
    for i in range(sample_times):
        prefix_shot = {s:"" for s in shots_value}
        data = json.load(open(file_shot,"r"))
        if level == "turn":
            prefixes = []
            for d in data:
                prefixes += shot_converter(sample=d,level=level)
            print(f"SHOTS LEN LIST: {len(prefixes)}")
            random.Random(i).shuffle(prefixes)
            for shots in prefix_shot.keys():
                prefix_shot[shots] = f"{shot_separator}".join(prefixes[:int(shots)]) + shot_separator
        else:
            random.Random(i).shuffle(data)
            shots = 0
            prefix = ""
            for d in data:
                prefix += shot_converter(sample=d,level=level) + shot_separator
                shots += 1
                if shots in prefix_shot:
                    prefix_shot[shots] = copy.copy(prefix)
        print(f"Loaded {name_dataset} {prefix_shot.keys()} shots for shuffle {i}!")
        prefix_list.append(prefix_shot)
    return prefix_list


def compute_ppl(model, tokenizer, device, prefix, query, max_seq, image_chat=False): 
    if image_chat:
        input_ids = tokenizer([prefix])
    else:
        input_ids = tokenizer([prefix+query])
        if len(input_ids['input_ids'][0])>max_seq:
            input_ids['input_ids'][0] = input_ids['input_ids'][0][-max_seq:]
            input_ids["attention_mask"][0] = input_ids["attention_mask"][0][-max_seq:]

    total_input_len = len(input_ids["input_ids"][0])
    query_tok_len = len(tokenizer([query])['input_ids'][0])
    label = torch.tensor([[-100]*(total_input_len-query_tok_len)+input_ids["input_ids"][0][-query_tok_len:]])
    input_ids["input_ids"] = torch.tensor(input_ids["input_ids"]).to(device)
    input_ids["attention_mask"] = torch.tensor(input_ids["attention_mask"]).to(device)
    with torch.no_grad():
        outputs = model(**input_ids, labels=label.to(device))

    return outputs.loss.item()         
            
def evalute_ppl(model, tokenizer, shot_converter, file_to_eval, 
                prefix, device, max_number_turns, level, max_seq,
                meta_type="all",verbose=False):
    loss_list = []
    for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
        if meta_type == "query":
            temp =  {"meta": dialogue["meta"], "dialogue": [], "query": []}
        elif meta_type == "incremental" or meta_type == "none":
            temp =  {"meta": [], "dialogue": [], "KG": []}
        elif meta_type == "linear":
            temp =  {"meta": None, "dialogue": [], "state": []}
        elif meta_type == "predict":
            temp =  {"meta": [], "dialogue": [], "state":[]}
        elif meta_type == "last_turn":
            dial = dialogue["dialogue"][:-1]
            dial.append([dialogue["dialogue"][-1][0],""])
            temp =  {"meta": dialogue["meta"], "dialogue": dial}
            if "meta_info" in dialogue:
                temp["meta_info"] = dialogue["meta_info"]
            prefix_plus_dial_history = prefix + shot_converter(sample=temp,level=level)+" "
            ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                              device=device, prefix=prefix_plus_dial_history, 
                              query=dialogue["dialogue"][-1][1], max_seq=max_seq)
            loss_list.append(ppl)
            if verbose:
                print('----'*10)
                print('----'*5+"PREFIX+DH"+'----'*5)
                print('----'*10)
                print(prefix_plus_dial_history)
                print('----'*10)
                print('----'*10)
                print('----'*5+"GOLD"+'----'*5)
                print('----'*10)
                print(dialogue["dialogue"][-1][1])
                print(f"PPL: {math.exp(ppl)}")
                print('----'*10)
                input()
                break
            continue
        else:
            print("Choose a meta-type")

        for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
            temp["dialogue"].append([user_utt,""])
            if meta_type == "incremental" or meta_type == "predict":
                if "KG" in dialogue:
                    temp["KG"].append(dialogue['KG'][id_t])
                    gold = "none" if len(dialogue['KG'][id_t])==0 else dialogue['KG'][id_t][0]
                else:
                    temp["meta"].append(dialogue['meta'][id_t])
                    gold = "none" if len(dialogue['meta'][id_t])==0 else dialogue['meta'][id_t][0]
            if meta_type == "query":
                temp["query"].append(dialogue['query'][id_t])
                gold = "none" if len(dialogue['query'][id_t])==0 else dialogue['query'][id_t][0]

            ## prepare input prefix
            ## NOTICE: the last space is needed beacuse of GPT tokenizer 
            prefix_plus_dial_history = prefix + shot_converter(sample=temp,level=level)+" "
            ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                              device=device, prefix=prefix_plus_dial_history, 
                              query=gold, max_seq=max_seq)
            loss_list.append(ppl)

            # add gold utterance into sys_utt
            temp["dialogue"][-1][1] = sys_utt
            temp["dialogue"] = temp["dialogue"][-max_number_turns:]
    
            if meta_type == "query":
                temp["query"] = temp["query"][-max_number_turns:]
                assert len(temp["dialogue"]) == len(temp["query"])

            if meta_type == "incremental" or meta_type == "predict":
                if "KG" in dialogue:
                    temp["KG"] = temp["KG"][-max_number_turns:]
                    assert len(temp["dialogue"]) == len(temp["KG"])
                else:  
                    temp["meta"] = temp["meta"][-max_number_turns:]
                    assert len(temp["dialogue"]) == len(temp["meta"])
        if verbose: break
    return math.exp(np.mean(loss_list))


def gen_continuation(tokenizer, model, device, multigpu, prefix_query, do_sample, eos_token_id, beam, gen_len, max_seq):
    input_ids = tokenizer(str(prefix_query), return_tensors='pt')
    input_len = len(input_ids['input_ids'][0])

    if multigpu: 
        with torch.no_grad():
            output = model.generate(
                **input_ids,
                do_sample=do_sample,
                max_length=input_len+gen_len if input_len+gen_len<max_seq else max_seq,
                eos_token_id=eos_token_id, # "\n"
                num_beams=beam,
                early_stopping=True,
            )
    else:
        with torch.no_grad():
            output = model.generate(
                input_ids = input_ids['input_ids'].to(device),
                do_sample=do_sample,
                max_length=input_len+gen_len if input_len+gen_len<max_seq else max_seq,
                eos_token_id=eos_token_id, # "\n"
                num_beams=beam,
                early_stopping=True,
            )
    response = tokenizer.decode(output[0][input_len:])
    response = response.split("\n")[0].strip()
    return response

### THIS IS SOME NASTY CODE FOR THE KG-PATH GENERATION
def retrive_nodes(trip):
    # print(title)
    try:
        if len(trip) == 1:
            sbj,rel, _ = trip[0]
            rel = rel.replace("-","_").replace("~","").replace(" ","_").lower()
            nodes = ks.run(f'MATCH (n1)-[r:{rel}]-(n2) WHERE n1.value= "{sbj}" RETURN n2.value as node').data()
        elif len(trip) == 2:
            sbj_1,rel_1, obj_1 = trip[0]
            sbj_2,rel_2, obj_2 = trip[1]
            assert sbj_2 == obj_1
            rel_1 = rel_1.replace("-","_").replace("~","").replace(" ","_").lower()
            rel_2 = rel_2.replace("-","_").replace("~","").replace(" ","_").lower()
            if rel_2!= "" and rel_1!="":
                nodes = ks.run(f'MATCH (n1)-[r:{rel_1}]-(n2)-[r2:{rel_2}]-(n3) WHERE n1.value= "{sbj_1}" and n2.value= "{obj_1}" RETURN n3.value as node').data()     
            else:
                return []  
        else:
            print("ERROR")
            exit(0)
        return [n["node"] for n in nodes]

    except:
        return [] 

def retrive_relation(trip):
    # print(title)
    try:
        if len(trip) == 1:
            sbj,_, _ = trip[0]
            relations = ks.run(f'MATCH (n1)-[r]-(n2) WHERE n1.value= "{sbj}" RETURN type(r) as relation').data()
        elif len(trip) == 2:
            sbj_1,rel_1, obj_1 = trip[0]
            sbj_2,rel_2, obj_2 = trip[1]
            # assert sbj_2 == obj_1
            rel_1 = rel_1.replace("-","_").replace("~","").replace(" ","_").lower()
            relations = ks.run(f'MATCH (n1)-[r:{rel_1}]-(n2)-[r2]-(n3) WHERE n1.value= "{sbj_1}" and n2.value= "{obj_1}" RETURN type(r2) as relation').data()       
        else:
            print("ERROR")
            exit(0)
        return [n["relation"] for n in relations]
    except:
        return [] 



def calculate_log_prob(model, tokenizer, prefix, targets):
    label2id = {}
    for target in targets:
        # works for single token label e.g., true or false, yes or no
        # label2id[target] = tokenizer.convert_tokens_to_ids(target)
        label2id[target] = tokenizer(target)["input_ids"][0] # only take the first token

    tokenized = tokenizer(prefix, return_tensors='pt')
    input_ids = tokenized.input_ids.cuda()
    attention_mask = tokenized.attention_mask.cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()[-1]
        prob = torch.nn.functional.softmax(logits, dim=-1)
        prob = prob.cpu().detach().numpy()
    normalized_scores = []

    for c in targets:
        score = prob[label2id[c]]
        normalized_scores.append([c,score])

    normalized_scores = sorted(normalized_scores, key=lambda x: x[1], reverse=True)
    normalized_scores = [[str(n),str(s)]for [n,s] in normalized_scores]
    return normalized_scores[0][0], normalized_scores

def get_triples_1(p):
    if "\t\t" in p:
        triple = p.split("\t\t")[0]
    else:
        triple = p
    if len(triple.split("\t")) > 2: 
        sbj_p, rel_p = triple.split("\t")[:2]
    elif len(triple.split("\t")) == 2: 
        sbj_p, rel_p = triple.split("\t")
    elif len(triple.split("\t")) == 1: 
        sbj_p = triple.split("\t")[0]    
        rel_p = ''
    else:
        sbj_p = ''
        rel_p = ''
    return [[sbj_p,rel_p,""]]

def get_triples_2(p,trip_list):
    
    triple = p.split("\t\t")[1]

    if len(triple.split("\t")) > 2: 
        sbj_p, rel_p = triple.split("\t")[:2]
    elif len(triple.split("\t")) == 2: 
        sbj_p, rel_p = triple.split("\t")
    elif len(triple.split("\t")) == 1: 
        sbj_p = triple.split("\t")[0]    
        rel_p = ''
    else:
        sbj_p = ''
        rel_p = ''
    trip_list.append([sbj_p,rel_p,""])
    return trip_list



def generate_response_DKG(model, tokenizer, shot_converter, file_to_eval, 
                      prefix, device, max_number_turns, level, 
                      meta_type="all", gen_len=50, beam=1,max_seq=1024, 
                      eos_token_id=198, do_sample=False, multigpu=False,verbose=False):
    results = []
    ud_id = 0
    for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
        ud_id += 1
        temp =  {"KG": [], "dialogue": []}

        res_temp =  {"KG": [], "dialogue": []}
        for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
            temp["dialogue"].append([user_utt,""])
            if meta_type == "incremental":
                temp["KG"].append([])

            
            prefix_query = prefix + shot_converter(sample=temp,level=level)
            if verbose:
                print('----'*10)
                print('----'*5+"PREFIX"+'----'*5)
                print('----'*10)
                print(prefix_query)
                print('----'*10)

            first_gen = gen_continuation(tokenizer, model, device, multigpu, prefix_query, do_sample, eos_token_id, beam, gen_len, max_seq)
        
            
            if "\t" in first_gen:
                triple = get_triples_1(first_gen)
                node_gen = retrive_nodes(triple)
                if len(node_gen)==0:
                    rel_gen = retrive_relation(triple)
                    rel_gen = list(dict.fromkeys(rel_gen))
                    if len(rel_gen)>0:
                        prefix_temp = prefix_query+ f" {triple[0][0]}\t"
                        next_rel, _ = calculate_log_prob(model,tokenizer,prefix_temp,rel_gen)
                        triple[0][1] = next_rel
                        node_gen = retrive_nodes(triple)
                        

                if len(node_gen):
                    prefix_new = prefix_query+ f" {triple[0][0]}\t{triple[0][1]}\t"
                    next_node, node_score = calculate_log_prob(model,tokenizer,prefix_new,node_gen)
                    triple[0][2] = next_node
                    prefix_new = prefix_new + next_node.strip()
                    
                    second_gen = gen_continuation(tokenizer, model, device, multigpu, prefix_new, do_sample, eos_token_id, beam, gen_len, max_seq)

                    if "\t\t" in second_gen:
                        triple = get_triples_2(second_gen,triple)
                        node_gen_2 = []
                        if triple[1][0] == triple[0][2]:
                            node_gen_2 = retrive_nodes(triple)
                        else:
                            triple[1][0] = triple[0][2]

                        if len(node_gen_2) == 0:
                            rel_gen = retrive_relation(triple)
                            rel_gen = list(dict.fromkeys(rel_gen))
                            if len(rel_gen)>0:
                                prefix_temp = prefix_new+ f"\t\t{triple[1][0]}\t"
                                # print(prefix_temp)
                                next_rel, rel_score = calculate_log_prob(model,tokenizer,prefix_temp,rel_gen)
                                # print(next_rel, rel_score)
                                triple[1][1] = next_rel
                                node_gen_2 = retrive_nodes(triple)
                                # print(node_gen_2)
                        node_gen_2 = list(filter(lambda node: node != triple[0][0], node_gen_2))
                        # print(node_gen_2)

                        prefix_new_2 = prefix_new+ f"\t\t{triple[1][0]}\t{triple[1][1]}\t"
                        next_node_2, node_score_2 = calculate_log_prob(model,tokenizer,prefix_new_2,node_gen_2)
                        response = [f"{triple[0][0]}\t{triple[0][1]}\t{triple[0][2]}\t\t{triple[1][0]}\t{triple[1][1]}\t{next_node_2}", node_score_2]
                    else:
                        response = [f"{triple[0][0]}\t{triple[0][1]}\t{next_node}", node_score]
                else:
                    response = ["None", []]
            else: 
                response = ["None", []]
            if verbose:
                print('----'*10)
                print('----'*5+"RESPONSE"+'----'*5)
                print('----'*10)
                print(response)
                print('----'*10)
                input()
            res_temp["KG"].append([response])

            temp["dialogue"][-1][1] = sys_utt
            temp["KG"][-1] = [] if response[0].lower() == "none" else [response[0]]
            temp["dialogue"] = temp["dialogue"][-max_number_turns:]
            temp["KG"] = temp["KG"][-max_number_turns:]
            assert len(temp["dialogue"]) == len(temp["KG"])

        results.append(res_temp)
        if verbose:
            break
    return results





def generate_response(model, tokenizer, shot_converter, file_to_eval, 
                      prefix, device, max_number_turns, level, 
                      meta_type="all", gen_len=50, beam=1,max_seq=1024, 
                      eos_token_id=198, do_sample=False, multigpu=False,verbose=False):
    results = []
    for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
        if meta_type == "query":
            temp =  {"meta": dialogue["meta"], "dialogue": [], "query": []}
        elif meta_type == "incremental" or meta_type == "none":
            temp =  {"meta": [], "dialogue": []}
        elif meta_type == "linear":
            temp =  {"meta": None, "dialogue": [], "state":["none"]}
        elif meta_type == "predict":
            temp =  {"meta": [], "dialogue": [], "state":[]}
        elif meta_type == "last_turn":
            dial = dialogue["dialogue"][:-1]
            dial.append([dialogue["dialogue"][-1][0],""])
            temp =  {"meta": dialogue["meta"], "dialogue": dial}
            if "meta_info" in dialogue:
                temp["meta_info"] = dialogue["meta_info"]

            prefix_query = prefix + shot_converter(sample=temp,level=level)
            if verbose:
                print('----'*10)
                print('----'*5+"PREFIX"+'----'*5)
                print('----'*10)
                print(prefix_query)
                print('----'*10)
            input_ids = tokenizer(str(prefix_query), return_tensors='pt')
            input_len = len(input_ids['input_ids'][0])

            if multigpu: 
                with torch.no_grad():
                    output = model.generate(
                        **input_ids,
                        do_sample=do_sample,
                        max_length=input_len+gen_len if input_len+gen_len<max_seq else max_seq,
                        eos_token_id=eos_token_id, # "\n"
                        num_beams=beam,
                        early_stopping=True,
                    )
            else:
                with torch.no_grad():
                    output = model.generate(
                        input_ids = input_ids['input_ids'].to(device),
                        do_sample=do_sample,
                        max_length=input_len+gen_len if input_len+gen_len<max_seq else max_seq,
                        eos_token_id=eos_token_id, # "\n"
                        num_beams=beam,
                        early_stopping=True,
                    )
            response = tokenizer.decode(output[0][input_len:])
            response = response.split("\n")[0].strip()
            if verbose:
                print('----'*10)
                print('----'*5+"RESPONSE"+'----'*5)
                print('----'*10)
                print(response)
                print('----'*10)
                input()
                break
            results.append({"meta": [[response]]})
            continue
        else:
            print("Choose a meta-type")

        res_temp =  {"meta": [], "dialogue": [], "state":[{}], "query":[]}
        for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
            temp["dialogue"].append([user_utt,""])
            if meta_type == "incremental":
                temp["meta"].append(dialogue['meta'][id_t])
            if meta_type == "query":
                temp["query"].append(dialogue['query'][id_t])
            if meta_type == "predict":
                temp["meta"].append([])
            
            prefix_query = prefix + shot_converter(sample=temp,level=level)
            if verbose:
                print('----'*10)
                print('----'*5+"PREFIX"+'----'*5)
                print('----'*10)
                print(prefix_query)
                print('----'*10)
            input_ids = tokenizer(str(prefix_query), return_tensors='pt')
            input_len = len(input_ids['input_ids'][0])

            if multigpu: 
                with torch.no_grad():
                    output = model.generate(
                        **input_ids,
                        do_sample=do_sample,
                        max_length=input_len+gen_len if input_len+gen_len<max_seq else max_seq,
                        eos_token_id=eos_token_id, # "\n"
                        num_beams=beam,
                        early_stopping=True,
                    )
            else:
                with torch.no_grad():
                    output = model.generate(
                        input_ids = input_ids['input_ids'].to(device),
                        do_sample=do_sample,
                        max_length=input_len+gen_len if input_len+gen_len<max_seq else max_seq,
                        eos_token_id=eos_token_id, # "\n"
                        num_beams=beam,
                        early_stopping=True,
                    )
            response = tokenizer.decode(output[0][input_len:])
            response = response.split("\n")[0].strip()
            if meta_type == "query":
                res_temp["query"].append([response])
            else:
                res_temp["meta"].append([response])
            if verbose:
                print('----'*10)
                print('----'*5+"RESPONSE"+'----'*5)
                print('----'*10)
                print(response)
                print('----'*10)
                input()
            temp["dialogue"][-1][1] = sys_utt
            temp["dialogue"] = temp["dialogue"][-max_number_turns:]

            ## THIS IS FOR DIALOGUE STATE TRACKING
            if meta_type == "predict":
                
                try:
                    state_pred = {sv.split("=")[0].replace("_"," ") : sv.split("=")[1] for sv in response.split("\t")}
                    if res_temp["state"][-1]: 
                        current_state = res_temp["state"][-1].copy()
                        current_state.update(state_pred)
                        res_temp["state"].append(current_state)
                    else: 
                        res_temp["state"].append(state_pred)
                except:
                    # print("parsing error: Copy previous state to new")
                    current_state = res_temp["state"][-1].copy()
                    res_temp["state"].append(current_state)


                state_string_li = []
                for slot, value in res_temp["state"][-1].items():
                    state_string_li.append(f"{slot.replace(' ','_')}={value}")
                temp["state"].append("\t".join(state_string_li))
                temp["meta"][-1] = [response]

            if meta_type == "query":
                temp["query"] = temp["query"][-max_number_turns:]
                assert len(temp["dialogue"]) == len(temp["query"])

            if meta_type == "incremental" or meta_type == "predict":
                temp["meta"] = temp["meta"][-max_number_turns:]
                assert len(temp["dialogue"]) == len(temp["meta"])
        results.append(res_temp)
        if verbose:
            break
    return results

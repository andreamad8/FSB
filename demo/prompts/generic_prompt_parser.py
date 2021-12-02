import torch
import math
import numpy as np
import json
from tqdm import tqdm
import logging
import copy
import random
logging.getLogger('transformers.generation_utils').setLevel(logging.CRITICAL)


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
                prefixes += shot_converter(sample=d)
            print(f"SHOTS LEN LIST: {len(prefixes)}")
            random.Random(i).shuffle(prefixes)
            for shots in prefix_shot.keys():
                prefix_shot[shots] = f"{shot_separator}".join(prefixes[:int(shots)]) + shot_separator
        else:
            random.Random(i).shuffle(data)
            shots = 0
            prefix = ""
            for d in data:
                prefix += shot_converter(sample=d) + shot_separator
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
    with torch.inference_mode():
        outputs = model(**input_ids, labels=label.to(device))

    return outputs.loss.item()         
            
def evalute_ppl(model, tokenizer, shot_converter, file_to_eval, 
                prefix, device, max_number_turns, level, max_seq,
                meta_type="all",verbose=False):
    loss_list = []
    for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
        if meta_type == "sentence":
            temp =  {"query": "", "dialogue": dialogue["dialogue"]}            
            prefix_query = prefix + shot_converter(sample=temp)+" "

            if type(dialogue["query"]) == str:
                query = dialogue["query"]
            elif len(dialogue["query"]) == 1:
                query = "\t".join(dialogue["query"][0])
            elif len(dialogue["query"]) == 2:
                query = "\t".join(dialogue["query"][0]) + "\t\t" + "\t".join(dialogue["query"][1]) 
            else:
                print("There might be an error")
                print(dialogue["query"])
                input()
            ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                              device=device, prefix=prefix_query, 
                              query=query, max_seq=max_seq)
            loss_list.append(ppl)
        elif meta_type == "sentencedynamic":
            temp =  {"query": "", "dialogue": dialogue["dialogue"]}
            prompt = ""
            for shot in dialogue["shots"][:prefix]:
                prompt += shot_converter(sample=shot) + "\n\n"  
            prefix_query = prompt + shot_converter(sample=temp)+" "

            # check if query is a string 
            if type(dialogue["query"]) == str:
                query = dialogue["query"]
            elif len(dialogue["query"]) == 1:
                query = "\t".join(dialogue["query"][0])
            elif len(dialogue["query"]) == 2:
                query = "\t".join(dialogue["query"][0]) + "\t\t" + "\t".join(dialogue["query"][1]) 
            else:
                print("There might be an error")
                print(dialogue["query"])
                input()
            ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                              device=device, prefix=prefix_query, 
                              query=query, max_seq=max_seq)
            loss_list.append(ppl)

    return math.exp(np.mean(loss_list))


def gen_continuation(tokenizer, model, device, multigpu, prefix_query, do_sample, eos_token_id, beam, gen_len, max_seq):
    input_ids = tokenizer(str(prefix_query), return_tensors='pt')
    input_len = len(input_ids['input_ids'][0])

    if multigpu: 
        with torch.inference_mode():
            output = model.generate(
                **input_ids,
                do_sample=do_sample,
                max_length=input_len+gen_len if input_len+gen_len<max_seq else max_seq,
                eos_token_id=eos_token_id, # "\n"
                num_beams=beam,
                early_stopping=True,
            )
    else:
        with torch.inference_mode():
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
def retrive_nodes(triple,KG):
    # print(title)
    try:
        if len(triple) == 1:
            sbj,rel, _ = triple[0]
            rel = rel.replace("-","_").replace("~","").replace(" ","_").lower()
            nodes = KG.run(f'MATCH (n1)-[r:{rel}]-(n2) WHERE n1.value= "{sbj}" RETURN n2.value as node').data()
        elif len(triple) == 2:
            sbj_1,rel_1, obj_1 = triple[0]
            sbj_2,rel_2, obj_2 = triple[1]
            assert sbj_2 == obj_1
            rel_1 = rel_1.replace("-","_").replace("~","").replace(" ","_").lower()
            rel_2 = rel_2.replace("-","_").replace("~","").replace(" ","_").lower()
            if rel_2!= "" and rel_1!="":
                nodes = KG.run(f'MATCH (n1)-[r:{rel_1}]-(n2)-[r2:{rel_2}]-(n3) WHERE n1.value= "{sbj_1}" and n2.value= "{obj_1}" RETURN n3.value as node').data()     
            else:
                return []  
        else:
            print("ERROR")
            exit(0)
        return [n["node"] for n in nodes]

    except:
        return [] 

def retrive_relation(triple,KG):
    # print(title)
    try:
        if len(triple) == 1:
            sbj,_, _ = triple[0]
            relations = KG.run(f'MATCH (n1)-[r]-(n2) WHERE n1.value= "{sbj}" RETURN type(r) as relation').data()
        elif len(triple) == 2:
            sbj_1,rel_1, obj_1 = triple[0]
            sbj_2,rel_2, obj_2 = triple[1]
            # assert sbj_2 == obj_1
            rel_1 = rel_1.replace("-","_").replace("~","").replace(" ","_").lower()
            relations = KG.run(f'MATCH (n1)-[r:{rel_1}]-(n2)-[r2]-(n3) WHERE n1.value= "{sbj_1}" and n2.value= "{obj_1}" RETURN type(r2) as relation').data()       
        else:
            print("ERROR")
            exit(0)
        return [n["relation"] for n in relations]
    except:
        return [] 



def calculate_log_prob(model, tokenizer, prefix, targets, device):
    label2id = {}
    for target in targets:
        # works for single token label e.g., true or false, yes or no
        # label2id[target] = tokenizer.convert_tokens_to_ids(target)
        label2id[target] = tokenizer(target)["input_ids"][0] # only take the first token

    tokenized = tokenizer(prefix, return_tensors='pt')
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()[-1]
        prob = torch.nn.functional.softmax(logits, dim=-1)
        prob = prob.cpu().detach().numpy()
    normalized_scores = []

    for c in targets:
        score = prob[label2id[c]]
        normalized_scores.append([c,score])

    normalized_scores = sorted(normalized_scores, key=lambda x: x[1], reverse=True)
    normalized_scores = [[str(n),str(s)] for [n,s] in normalized_scores]
    if len(normalized_scores) > 0:
        return normalized_scores[0][0], normalized_scores
    else:
        return "", [["",'0.0']]


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


def get_prompt(dialogue,prefix,shot_converter):
    prompt = ""
    for shot in dialogue["shots"][:prefix]:
        prompt += shot_converter(sample=shot) + "\n\n"
    return prompt

def generate_response_DKG(model, tokenizer, shot_converter, file_to_eval, 
                      prefix, device, max_number_turns, level, 
                      meta_type="all", gen_len=50, beam=1,max_seq=1024, 
                      eos_token_id=198, do_sample=False, multigpu=False,
                      verbose=False, KG=None):
    results = []
    for dialogue in tqdm(json.load(open(file_to_eval,"r"))):

        if meta_type == "sentencedynamic":
            temp =  {"query": "", "dialogue": dialogue["dialogue"]}
            prompt = get_prompt(dialogue,prefix,shot_converter)
            prefix_query = prompt + shot_converter(sample=temp)

            # check if the prefix is too long
            input_ids = tokenizer(prefix_query, return_tensors='pt')
            input_len = len(input_ids['input_ids'][0])
            shot_number_temp = prefix
            while input_len + gen_len > max_seq-200:
                shot_number_temp = shot_number_temp - 1
                print(f"Prefix too long, decrease shot number from {prefix} to {shot_number_temp}")
                prompt = get_prompt(dialogue,shot_number_temp,shot_converter)
                prefix_query = prompt + shot_converter(sample=temp)

                # check if the prefix is too long
                input_ids = tokenizer(prefix_query, return_tensors='pt')
                input_len = len(input_ids['input_ids'][0])
        else:
            temp =  {"query": "", "dialogue": dialogue["dialogue"]}            
            prefix_query = prefix + shot_converter(sample=temp)
            if verbose:
                print('----'*10)
                print('----'*5+"PREFIX"+'----'*5)
                print('----'*10)
                print(prefix_query)
                print('----'*10)

        first_gen = gen_continuation(tokenizer, model, device, multigpu, prefix_query, do_sample, eos_token_id, beam, gen_len, max_seq)
    
        
        if "\t" in first_gen:
            triple = get_triples_1(first_gen)
            node_gen = retrive_nodes(triple,KG)
            if len(node_gen)==0:
                rel_gen = retrive_relation(triple,KG)
                rel_gen = list(dict.fromkeys(rel_gen))
                if len(rel_gen)>0:
                    prefix_temp = prefix_query+ f" {triple[0][0]}\t"
                    next_rel, _ = calculate_log_prob(model,tokenizer,prefix_temp,rel_gen,device)
                    triple[0][1] = next_rel
                    node_gen = retrive_nodes(triple,KG)
                    

            if len(node_gen):
                prefix_new = prefix_query+ f" {triple[0][0]}\t{triple[0][1]}\t"
                next_node, node_score = calculate_log_prob(model,tokenizer,prefix_new,node_gen,device)
                triple[0][2] = next_node
                prefix_new = prefix_new + next_node.strip()
                
                second_gen = gen_continuation(tokenizer, model, device, multigpu, prefix_new, do_sample, eos_token_id, beam, gen_len, max_seq)

                if "\t\t" in second_gen:
                    triple = get_triples_2(second_gen,triple)
                    node_gen_2 = []
                    if triple[1][0] == triple[0][2]:
                        node_gen_2 = retrive_nodes(triple,KG)
                    else:
                        triple[1][0] = triple[0][2]

                    if len(node_gen_2) == 0:
                        rel_gen = retrive_relation(triple,KG)
                        rel_gen = list(dict.fromkeys(rel_gen))
                        if len(rel_gen)>0:
                            prefix_temp = prefix_new+ f"\t\t{triple[1][0]}\t"
                            # print(prefix_temp)
                            next_rel, rel_score = calculate_log_prob(model,tokenizer,prefix_temp,rel_gen,device)
                            # print(next_rel, rel_score)
                            triple[1][1] = next_rel
                            node_gen_2 = retrive_nodes(triple,KG)
                            # print(node_gen_2)
                    node_gen_2 = list(filter(lambda node: node != triple[0][0], node_gen_2))
                    # print(node_gen_2)

                    prefix_new_2 = prefix_new+ f"\t\t{triple[1][0]}\t{triple[1][1]}\t"
                    next_node_2, node_score_2 = calculate_log_prob(model,tokenizer,prefix_new_2,node_gen_2,device)
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
        results.append({"query": response})
        if verbose:
            break
    return results


def get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu):
    input_ids = tokenizer(str(prefix_query), return_tensors='pt')
    input_len = len(input_ids['input_ids'][0])
    # remove the first 100 characters from the prefix
    if input_len + gen_len > max_seq-200:
        print("WARNING: the prefix is too long, truncating it") 
        print(f"Tokenized length: {input_len}")
        token_to_remove = input_len + gen_len - (max_seq - 200)
        input_ids['input_ids'] = input_ids['input_ids'][:,token_to_remove:]
        input_len = len(input_ids['input_ids'][0])
        print(f"New Tokenized length: {input_len}")
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

def generate_response(model, tokenizer, shot_converter, file_to_eval, 
                      prefix, device, max_number_turns, level, 
                      meta_type="all", gen_len=50, beam=1,max_seq=1024, 
                      eos_token_id=198, do_sample=False, multigpu=False,verbose=False):
    results = []
    for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
        if meta_type == "sentencedynamic":
            temp =  {"query": "", "dialogue": dialogue["dialogue"]}
            prompt = get_prompt(dialogue,prefix,shot_converter)
            prefix_query = prompt + shot_converter(sample=temp)

            # check if the prefix is too long
            input_ids = tokenizer(prefix_query, return_tensors='pt')
            input_len = len(input_ids['input_ids'][0])
            shot_number_temp = prefix
            while input_len + gen_len > max_seq-100:
                shot_number_temp = shot_number_temp - 1
                print(f"Prefix too long, decrease shot number from {prefix} to {shot_number_temp}")
                prompt = get_prompt(dialogue,shot_number_temp,shot_converter)
                prefix_query = prompt + shot_converter(sample=temp)

                # check if the prefix is too long
                input_ids = tokenizer(prefix_query, return_tensors='pt')
                input_len = len(input_ids['input_ids'][0])
        elif meta_type == "sentence":
            temp =  {"query": "", "dialogue": dialogue["dialogue"]}            
            prefix_query = prefix + shot_converter(sample=temp)
            if verbose:
                print('----'*10)
                print('----'*5+"PREFIX"+'----'*5)
                print('----'*10)
                print(prefix_query)
                print('----'*10)
        else:
            print("ERROR: meta_type not recognized")
            exit(1)

        response = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu)
        if verbose:
            print('----'*10)
            print('----'*5+"RESPONSE"+'----'*5)
            print('----'*10)
            print(response)
            print('----'*10)
            input()
            break
        results.append({"query": response})
    return results


def generate_response_DKG_interactive(model, tokenizer, shot_converter, dialogue, 
                      prefix, device, level, gen_len=50, beam=1,max_seq=1024, 
                      eos_token_id=198, do_sample=False, multigpu=False, verbose=False, KG=None):
        

            
    prefix_query = prefix + shot_converter(sample=dialogue)

    first_gen = gen_continuation(tokenizer, model, device, multigpu, prefix_query, do_sample, eos_token_id, beam, gen_len, max_seq)

    
    if "\t" in first_gen:
        triple = get_triples_1(first_gen)
        node_gen = retrive_nodes(triple,KG)
        if len(node_gen)==0:
            rel_gen = retrive_relation(triple,KG)
            rel_gen = list(dict.fromkeys(rel_gen))
            if len(rel_gen)>0:
                prefix_temp = prefix_query+ f" {triple[0][0]}\t"
                next_rel, _ = calculate_log_prob(model,tokenizer,prefix_temp,rel_gen,device)
                triple[0][1] = next_rel
                node_gen = retrive_nodes(triple,KG)
                

        if len(node_gen):
            prefix_new = prefix_query+ f" {triple[0][0]}\t{triple[0][1]}\t"
            next_node, node_score = calculate_log_prob(model,tokenizer,prefix_new,node_gen,device)
            triple[0][2] = next_node
            prefix_new = prefix_new + next_node.strip()
            
            second_gen = gen_continuation(tokenizer, model, device, multigpu, prefix_new, do_sample, eos_token_id, beam, gen_len, max_seq)

            if "\t\t" in second_gen:
                triple = get_triples_2(second_gen,triple)
                node_gen_2 = []
                if triple[1][0] == triple[0][2]:
                    node_gen_2 = retrive_nodes(triple,KG)
                else:
                    triple[1][0] = triple[0][2]

                if len(node_gen_2) == 0:
                    rel_gen = retrive_relation(triple,KG)
                    rel_gen = list(dict.fromkeys(rel_gen))
                    if len(rel_gen)>0:
                        prefix_temp = prefix_new+ f"\t\t{triple[1][0]}\t"
                        # print(prefix_temp)
                        next_rel, rel_score = calculate_log_prob(model,tokenizer,prefix_temp,rel_gen,device)
                        # print(next_rel, rel_score)
                        triple[1][1] = next_rel
                        node_gen_2 = retrive_nodes(triple,KG)
                        # print(node_gen_2)
                node_gen_2 = list(filter(lambda node: node != triple[0][0], node_gen_2))
                # print(node_gen_2)

                prefix_new_2 = prefix_new+ f"\t\t{triple[1][0]}\t{triple[1][1]}\t"
                next_node_2, node_score_2 = calculate_log_prob(model,tokenizer,prefix_new_2,node_gen_2,device)
                response = f"{triple[0][0]}\t{triple[0][1]}\t{triple[0][2]}\t\t{triple[1][0]}\t{triple[1][1]}\t{next_node_2}"
            else:
                response = f"{triple[0][0]}\t{triple[0][1]}\t{next_node}"
        else:
            response = "None"
    else: 
        response = "None"

    return response
import torch
import math
import numpy as np
import json
from tqdm import tqdm
import logging
import copy
import random
from collections import defaultdict
import glob
import requests
logging.getLogger('transformers.generation_utils').setLevel(logging.CRITICAL)



def load_prefix(tokenizer, shots_value, shot_converter, 
                file_shot, name_dataset, with_knowledge, 
                shot_separator="\n\n",sample_times=5):
    prefix_list = []
    for i in range(sample_times):
        shots = 0
        prefix_shot = {s:"" for s in shots_value}
        data = json.load(open(file_shot,"r"))
        random.Random(i).shuffle(data)
        if "data/smd/weather-" in file_shot or "data/smd/navigate-" in file_shot:
            for shot in prefix_shot.keys():
                if shot != 0:
                    prefix_shot[shot] = shot_converter(sample=data[0],with_knowledge=shot) 
        else:
            prefix = ""
            for d in data:
                prefix += shot_converter(sample=d) + shot_separator
                shots += 1
                if shots in prefix_shot:
                    prefix_shot[shots] = copy.copy(prefix)
        print(f"Loaded {name_dataset} {prefix_shot.keys()} shots for shuffle {i}!")
        prefix_list.append(prefix_shot)
    return prefix_list


def load_prefix_by_category(tokenizer, shots_value, shot_converter, 
                file_shot, name_dataset, with_knowledge, 
                shot_separator="\n\n",sample_times=5):
    split = file_shot.replace(".json","")
    prefix_list = []
    for i in range(sample_times):
        prefix_shot_by_category = {}
        for file_shot_ in glob.glob(f"{split}/*_2.json"):
            shots = 0
            prefix_shot = {s:"" for s in shots_value}
            prefix = ""
            data = json.load(open(file_shot_,"r"))
            random.Random(i).shuffle(data)
            for d in data:
                prefix += shot_converter(sample=d) + shot_separator
                shots += 1
                if shots in prefix_shot:
                    prefix_shot[shots] = copy.copy(prefix)
            name_category = file_shot_.replace(".json","").replace(f"{split}/","")
            prefix_shot_by_category[name_category] = prefix_shot
        # print(prefix_shot_by_category[name_category])
        print(f"Loaded IC {len(prefix_shot_by_category.keys())} categories shots for shuffle {i}!")
        prefix_list.append(prefix_shot_by_category)

    return prefix_list



def compute_ppl(model, tokenizer, device, prefix, query, max_seq, image_chat=False, verbose=False): 
    if image_chat:
        input_ids = tokenizer([prefix])
    else:
        if verbose:
            print(prefix+query)
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
                prefix, device, max_number_turns, with_knowledge, max_seq,
                meta_type="all",verbose=False):
    if "all_turns" in meta_type:
        loss_list = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all_turns_category":
                temp =  {"personalities": [], "dialogue": []}
            else:
                temp =  {"meta": [], "dialogue": []}
    
            if "img" in dialogue:   
                temp["img"] = dialogue["img"]
                temp["personalities"] = []

            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append(["",""])

                if meta_type == "all_turns_category" or "img" in temp:
                    temp["personalities"].append(dialogue["personalities"][id_t])

                ## prepare input prefix
                ## NOTICE: the last space is needed beacuse of GPT tokenizer 
                if meta_type == "all_turns_category" and id_t == 0:
                    pass
                else:
                    prefix_plus_dial_history = prefix + shot_converter(sample=temp)+" "
                    ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                                    device=device, prefix=prefix_plus_dial_history, 
                                    query=user_utt, max_seq=max_seq)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX+DH"+'----'*5)
                        print('----'*10)
                        print(prefix_plus_dial_history)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"GOLD"+'----'*5)
                        print('----'*10)
                        print(user_utt)
                        print(f"PPL: {math.exp(ppl)}")

                        print('----'*10)
                        input()
                    loss_list.append(ppl)

                temp["dialogue"][-1][0] = user_utt

                if sys_utt == "" or meta_type == "all_turns_category":
                    pass
                else:

                    prefix_plus_dial_history = prefix + shot_converter(sample=temp)+" "
                    ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                                    device=device, prefix=prefix_plus_dial_history, 
                                    query=sys_utt, max_seq=max_seq)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX+DH"+'----'*5)
                        print('----'*10)
                        print(prefix_plus_dial_history)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"GOLD"+'----'*5)
                        print('----'*10)
                        print(sys_utt)
                        print(f"PPL: {math.exp(ppl)}")
                        print('----'*10)
                        input()
                    loss_list.append(ppl)
                    

                # add gold utterance into sys_utt
                temp["dialogue"][-1][1] = sys_utt
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]
            if verbose: break
        return math.exp(np.mean(loss_list))
    else:
        loss_list = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all":
                temp =  {"meta": dialogue["meta"], "dialogue": []}
            elif meta_type == "incremental" or meta_type == "none":
                temp =  {"meta": [], "dialogue": [], "KB": []}
            elif meta_type == "linear":
                temp =  {"meta": None, "dialogue": []}
            else:
                print("Choose a meta-type")

            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append([user_utt,""])
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"].append(dialogue['KB'][id_t])
                        temp["meta"] = dialogue["meta"]
                    else:
                        temp["meta"].append(dialogue['meta'][id_t])
                ## prepare input prefix
                ## NOTICE: the last space is needed beacuse of GPT tokenizer 
                prefix_plus_dial_history = prefix + shot_converter(sample=temp)+" "
                ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                                device=device, prefix=prefix_plus_dial_history, 
                                query=sys_utt, max_seq=max_seq)


                if verbose:
                    print('----'*10)
                    print('----'*5+"PREFIX+DH"+'----'*5)
                    print('----'*10)
                    print(prefix_plus_dial_history)
                    print('----'*10)
                    print('----'*10)
                    print('----'*5+"GOLD"+'----'*5)
                    print('----'*10)
                    print(sys_utt)
                    print(f"PPL: {math.exp(ppl)}")
                    print('----'*10)
                    input()
                loss_list.append(ppl)

                # add gold utterance into sys_utt
                temp["dialogue"][-1][1] = sys_utt
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"] = temp["KB"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["KB"])
                    else:
                        temp["meta"] = temp["meta"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["meta"])
            if verbose: break
        return math.exp(np.mean(loss_list))


def get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu):
    input_ids = tokenizer(str(prefix_query), return_tensors='pt')
    input_len = len(input_ids['input_ids'][0])
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
    

def get_prompt(dialogue,prefix,shot_converter):
    prompt = ""
    for shot in dialogue["shots"][:prefix]:
        prompt += shot_converter(sample=shot) + "\n\n"
    return prompt

def generate_response(model, tokenizer, shot_converter, file_to_eval, 
                      prefix, device, max_number_turns, with_knowledge, 
                      meta_type="all", gen_len=50, beam=1,max_seq=1024, 
                      eos_token_id=198, do_sample=False, multigpu=False, verbose=False):

    if "all_turns" in meta_type:
        results = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all_turns_category":
                temp =  {"personalities": [], "dialogue": []}
            else:
                temp =  {"meta": [], "dialogue": []}
            if "img" in dialogue:   
                temp["img"] = dialogue["img"]
                temp["personalities"] = []
            
            res_temp =  {"meta":dialogue["meta"] , "dialogue": []}

            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append(["",""])
                if meta_type == "all_turns_category" or "img" in temp:
                    temp["personalities"].append(dialogue["personalities"][id_t])

                if meta_type == "all_turns_category" and id_t == 0:
                    response_USR_A = ""
                    pass
                else:
                    prefix_query = prefix + shot_converter(sample=temp)
                    response_USR_A = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX"+'----'*5)
                        print('----'*10)
                        print(prefix_query)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"RESPONSE"+'----'*5)
                        print('----'*10)
                        print(response_USR_A)
                        print('----'*10)
                        input()

                temp["dialogue"][-1][0] = user_utt

                if sys_utt == "" or meta_type == "all_turns_category":
                    response_USR_B = ""
                    pass
                else:
                    prefix_query = prefix + shot_converter(sample=temp)
                    response_USR_B = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX"+'----'*5)
                        print('----'*10)
                        print(prefix_query)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"RESPONSE"+'----'*5)
                        print('----'*10)
                        print(response_USR_B)
                        print('----'*10)
                        input()
                temp["dialogue"][-1][1] = sys_utt

                res_temp["dialogue"].append([response_USR_A,response_USR_B])
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]

            results.append(res_temp)
            if verbose: 
                break
        return results
    else:
        results = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all":
                temp =  {"meta": dialogue["meta"], "dialogue": []}
            elif meta_type == "incremental" or meta_type == "none":
                temp =  {"meta": [], "dialogue": [], "KB": []}
            else:
                print("Choose a meta-type")

            res_temp =  {"meta": [], "dialogue": []}
            if "id" in dialogue:
                res_temp["id"] = dialogue["id"]
            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append([user_utt,""])
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"].append(dialogue['KB'][id_t])
                        temp["meta"] = dialogue["meta"]
                    else:
                        temp["meta"].append(dialogue['meta'][id_t])
                prefix_query = prefix + shot_converter(sample=temp)
                if verbose:
                    print('----'*10)
                    print('----'*5+"PREFIX"+'----'*5)
                    print('----'*10)
                    print(prefix_query)
                    print('----'*10)

                response = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu)
                res_temp["dialogue"].append([response])
                if verbose:
                    print('----'*10)
                    print('----'*5+"RESPONSE"+'----'*5)
                    print('----'*10)
                    print(response)
                    print('----'*10)
                    input()
                temp["dialogue"][-1][1] = sys_utt
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"] = temp["KB"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["KB"])
                    else:
                        temp["meta"] = temp["meta"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["meta"])
            results.append(res_temp)
            if verbose:
                break
        return results


def generate_response_dynamic(model, tokenizer, shot_converter, file_to_eval, 
                      prefix, device, max_number_turns, with_knowledge, 
                      meta_type="all", gen_len=50, beam=1,max_seq=1024, 
                      eos_token_id=198, do_sample=False, multigpu=False, verbose=False):

    if "all_turns" in meta_type:
        results = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all_turns_category":
                temp =  {"personalities": [], "dialogue": []}
            else:
                temp =  {"meta": [], "dialogue": []}
            if "img" in dialogue:   
                temp["img"] = dialogue["img"]
                temp["personalities"] = []
            
            res_temp =  {"meta":dialogue["meta"] , "dialogue": []}

            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append(["",""])
                if meta_type == "all_turns_category" or "img" in temp:
                    temp["personalities"].append(dialogue["personalities"][id_t])

                if meta_type == "all_turns_category" and id_t == 0:
                    response_USR_A = ""
                    pass
                else:
                    prefix_query = prefix + shot_converter(sample=temp)
                    response_USR_A = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX"+'----'*5)
                        print('----'*10)
                        print(prefix_query)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"RESPONSE"+'----'*5)
                        print('----'*10)
                        print(response_USR_A)
                        print('----'*10)
                        input()

                temp["dialogue"][-1][0] = user_utt

                if sys_utt == "" or meta_type == "all_turns_category":
                    response_USR_B = ""
                    pass
                else:
                    prefix_query = prefix + shot_converter(sample=temp)
                    response_USR_B = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX"+'----'*5)
                        print('----'*10)
                        print(prefix_query)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"RESPONSE"+'----'*5)
                        print('----'*10)
                        print(response_USR_B)
                        print('----'*10)
                        input()
                temp["dialogue"][-1][1] = sys_utt

                res_temp["dialogue"].append([response_USR_A,response_USR_B])
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]

            results.append(res_temp)
            if verbose: 
                break
        return results
    else:
        results = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all":
                temp =  {"meta": dialogue["meta"], "dialogue": []}
            elif meta_type == "incremental" or meta_type == "none":
                temp =  {"meta": [], "dialogue": [], "KB": []}
            else:
                print("Choose a meta-type")

            res_temp =  {"meta": [], "dialogue": []}
            if "id" in dialogue:
                res_temp["id"] = dialogue["id"]
            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append([user_utt,""])
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"].append(dialogue['KB'][id_t])
                        temp["meta"] = dialogue["meta"]
                    else:
                        temp["meta"].append(dialogue['meta'][id_t])
                prefix_query = prefix + shot_converter(sample=temp)
                if verbose:
                    print('----'*10)
                    print('----'*5+"PREFIX"+'----'*5)
                    print('----'*10)
                    print(prefix_query)
                    print('----'*10)

                response = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu)
                res_temp["dialogue"].append([response])
                if verbose:
                    print('----'*10)
                    print('----'*5+"RESPONSE"+'----'*5)
                    print('----'*10)
                    print(response)
                    print('----'*10)
                    input()
                temp["dialogue"][-1][1] = sys_utt
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"] = temp["KB"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["KB"])
                    else:
                        temp["meta"] = temp["meta"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["meta"])
            results.append(res_temp)
            if verbose:
                break
        return results



def evalute_prompt_prob(model, tokenizer, shot_converter, file_to_eval, 
                prefix, device, max_number_turns, with_knowledge, max_seq,
                meta_type="all",verbose=False, max_shot=1, repetition=0):

    loss_list = []
    id_dial = 0
    for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
        id_dial += 1
        if id_dial == 101: break
        temp = {"dialogue": []}
        for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
            temp["dialogue"].append([user_utt,""])

            prompt_ppl = defaultdict()
            for name, prompt in prefix.items():
                query = shot_converter(sample=temp)
                ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                                device=device, prefix=prompt[repetition][max_shot] + " ", 
                                query=query, 
                                max_seq=max_seq)

                prompt_ppl[name] = math.exp(ppl)


            loss_list.append(prompt_ppl)
            # add gold utterance into sys_utt
            temp["dialogue"][-1][1] = sys_utt
            temp["dialogue"] = temp["dialogue"][-max_number_turns:]

        if verbose: break
    return loss_list


def select_prompt_interactive(model, tokenizer, shot_converter, dialogue, 
                prompt_dict, device, max_seq, max_shot=1, sample=False):
    temp = {}
    temp["dialogue"] = dialogue["dialogue"][-2:]
    query = shot_converter(sample=temp, with_knowledge=None)
    prompt_ppl = defaultdict()
    for name, prompt in prompt_dict.items():
        ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                        device=device, prefix=prompt[max_shot], 
                        query=query,  max_seq=max_seq, verbose=False)
        prompt_ppl[name] = math.exp(ppl)
    if sample:
        sum_val = sum(prompt_ppl.values())
        prob_dict = {}
        for k, v in prompt_ppl.items():
            prob_dict[k] = sum_val-v
        return random.choices(list(prob_dict.keys()), weights=prob_dict.values(), k=1)[0]
    else:
        return min(prompt_ppl, key=prompt_ppl.get), prompt_ppl


def generate_response_interactive(model, tokenizer, shot_converter, dialogue, 
                      prefix, device, with_knowledge, 
                      meta_type="all", gen_len=50, beam=1,max_seq=1024, 
                      eos_token_id=198, do_sample=False, multigpu=False, 
                      api=False, api_key="", temperature=0.7, topp=0.9):


    prefix_query = prefix + shot_converter(dialogue, with_knowledge)
    input_ids = tokenizer(str(prefix_query), return_tensors='pt')
    input_len = len(input_ids['input_ids'][0])

    if api:
        response = requests.post(
            "https://api.ai21.com/studio/v1/j1-jumbo/complete",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "prompt": prefix_query, 
                "numResults": 1, 
                "maxTokens": input_len+gen_len if input_len+gen_len<max_seq else max_seq, 
                "stopSequences": ["\n"],
                "topP": topp,
                "temperature": temperature,
            }
        )
        json_data = json.loads(response.text)
        output = json_data['completions'][0]['data']['text']
        return output.split("\n")[0].strip()
    else:
        with torch.no_grad():
            output = model.generate(
                input_ids = input_ids['input_ids'].to(device),
                do_sample=do_sample,
                max_length=input_len+gen_len if input_len+gen_len<max_seq else max_seq,
                eos_token_id=eos_token_id, # "\n"
                num_beams=beam,
                early_stopping=True,
                top_p=topp,
                temperature=temperature,
                min_length=4,
                sample=True,
            )
    response = tokenizer.decode(output[0][input_len:])
    response = response.split("\n")[0].strip()
    return response
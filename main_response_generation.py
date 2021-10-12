import json
import os
import argparse
import numpy as np
from prompts.generic_prompt import load_prefix, load_prefix_by_category, evalute_ppl, generate_response
from prompts.persona_chat import convert_sample_to_shot_persona
from prompts.persona_chat_memory import convert_sample_to_shot_msc
from prompts.wizard_of_wikipedia import convert_sample_to_shot_wow
from prompts.wizard_of_internet import convert_sample_to_shot_wit
from prompts.emphatetic_dialogue import convert_sample_to_shot_ed
from prompts.dialKG import convert_sample_to_shot_dialKG
from prompts.daily_dialogue import convert_sample_to_shot_DD_prefix, convert_sample_to_shot_DD_inference
from prompts.image_chat import convert_sample_to_shot_IC_prefix, convert_sample_to_shot_IC_inference
from prompts.image_chat_with_img import convert_sample_to_shot_IC_img_prefix, convert_sample_to_shot_IC_img_inference
from prompts.smd import convert_sample_to_shot_smd, convert_sample_to_shot_smd_custum
from tabulate import tabulate
from metric.scorer import score
from collections import defaultdict
import os
from tqdm import tqdm
import glob
from utils.utils import load_model, save_file, checker_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"

mapper = {
          "persona": {"shot_converter":convert_sample_to_shot_persona, 
                    "shot_converter_inference": convert_sample_to_shot_persona,
                     "file_data":"data/persona/","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,5]},"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
        "smd-navigate": {"shot_converter":convert_sample_to_shot_smd_custum, 
                    "shot_converter_inference": convert_sample_to_shot_smd,
                     "file_data":"data/smd/navigate-","with_knowledge":None,
                     "shots":{1024:None,2048:[0,1,8]},"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
        "smd-schedule": {"shot_converter":convert_sample_to_shot_smd, 
                    "shot_converter_inference": convert_sample_to_shot_smd,
                     "file_data":"data/smd/schedule-","with_knowledge":None,
                     "shots":{1024:None,2048:[0,1,8]},"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
        "smd-weather": {"shot_converter":convert_sample_to_shot_smd_custum, 
                    "shot_converter_inference": convert_sample_to_shot_smd,
                     "file_data":"data/smd/weather-","with_knowledge":None,
                     "shots":{1024:None,2048:[0,1,8]},"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
          "msc-dialogue-2": {"shot_converter":convert_sample_to_shot_msc, 
                    "shot_converter_inference": convert_sample_to_shot_msc,
                     "file_data":"data/msc/session-2-","with_knowledge":None,
                     "shots":{1024:[0,1],2048:[0,1,3]},"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":3},
          "wow": {"shot_converter":convert_sample_to_shot_wow, 
                 "shot_converter_inference": convert_sample_to_shot_wow,
                 "file_data":"data/wow/","with_knowledge":True,
                  "shots":{1024:[0,1,2],2048:[0,1,2,3,4]},"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":5},
          "wit": {"shot_converter":convert_sample_to_shot_wit, 
                 "shot_converter_inference": convert_sample_to_shot_wit,
                 "file_data":"data/wit/","with_knowledge":True,
                  "shots":{1024:[0,1],2048:[0,1,2,3]},"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":60,"max_number_turns":4},
          "ed": {"shot_converter":convert_sample_to_shot_ed, 
                 "shot_converter_inference": convert_sample_to_shot_ed,
                 "file_data":"data/ed/","with_knowledge":None,
                  "shots":{1024:[0,1,7],2048:[0,1,17]},"shot_separator":"\n\n",
                  "meta_type":"none","gen_len":50,"max_number_turns":5},
          "dialKG": {"shot_converter":convert_sample_to_shot_dialKG, 
                 "shot_converter_inference": convert_sample_to_shot_dialKG,
                 "file_data":"data/dialKG/","with_knowledge":True,
                  "shots":{1024:[0,1,3],2048:[0,1,9]},"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":4},
          "DD": {"shot_converter":convert_sample_to_shot_DD_prefix, 
                 "shot_converter_inference": convert_sample_to_shot_DD_inference,
                 "file_data":"data/dailydialog/","with_knowledge":False,
                  "shots":{1024:[0,1,2],2048:[0,1,6]},"shot_separator":"\n\n",
                  "meta_type":"all_turns","gen_len":50,"max_number_turns":5},
          "IC": {"shot_converter":convert_sample_to_shot_IC_prefix, 
                 "shot_converter_inference": convert_sample_to_shot_IC_inference,
                 "file_data":"data/image_chat/","with_knowledge":False,
                  "shots":{1024:[0,1,5],2048:[0,1,10]},"shot_separator":"\n\n",
                  "meta_type":"all_turns_category","gen_len":50,"max_number_turns":5},
          "IC-img": {"shot_converter":convert_sample_to_shot_IC_img_prefix, 
                 "shot_converter_inference": convert_sample_to_shot_IC_img_inference,
                 "file_data":"data/image_chat/img_","with_knowledge":False,
                  "shots":{1024:[0,1,4],2048:[0,1,10]},"shot_separator":"\n\n",
                  "meta_type":"all_turns","gen_len":50,"max_number_turns":5},
         }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", default="gpt2",type=str,required=True)
    parser.add_argument("--dataset", default="persona",type=str,required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--sample_times", type=int, default=3)
    parser.add_argument("--do_sample", action='store_true', help="sample n times and rescore based on ppl")
    parser.add_argument("--multigpu", action='store_true', help="run on multiple gpus")
    parser.add_argument("--verbose", action='store_true', help="run on multiple gpus")
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    beam = args.beam
    model_checkpoint = args.model_checkpoint

    model, tokenizer, max_seq = load_model(args,model_checkpoint,device)
    
    list_of_dataset = args.dataset.split(",")
    for d in list_of_dataset:
        print(f"EVALUATING DATASET {d} on {model_checkpoint} with beam size {beam}")
        if "category" in mapper[d]['meta_type']:
            prefix_list = load_prefix_by_category(tokenizer=tokenizer, shots_value=mapper[d]["shots"][max_seq], 
                                shot_converter=mapper[d]["shot_converter"], 
                                file_shot=mapper[d]["file_data"]+"valid.json", 
                                name_dataset=d, with_knowledge=mapper[d]["with_knowledge"], 
                                shot_separator=mapper[d]["shot_separator"],sample_times=args.sample_times)

            for id_prefix, prefix_shot_by_category in enumerate(prefix_list):
                shot_results = defaultdict(lambda: defaultdict(list))
                for cat, prefix_shots in tqdm(prefix_shot_by_category.items()):

                    for shots, prefix in prefix_shots.items():
                        if checker_file(f"{d}_{shots}_{model_checkpoint}_{beam}-{args.do_sample}_{id_prefix}.json") or args.verbose:
                            shot_results[shots]["generated_out"] += generate_response(model, tokenizer, shot_converter=mapper[d]["shot_converter_inference"], 
                                                            file_to_eval=mapper[d]["file_data"]+f"test/{cat}.json", prefix=prefix, 
                                                            device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                            with_knowledge=mapper[d]["with_knowledge"], 
                                                            meta_type=mapper[d]["meta_type"], gen_len=mapper[d]["gen_len"], 
                                                            beam=beam, max_seq=max_seq, eos_token_id=198, 
                                                            do_sample=args.do_sample, multigpu=args.multigpu)

                            ppl_score = evalute_ppl(model, tokenizer, shot_converter=mapper[d]["shot_converter_inference"], 
                                                    file_to_eval=mapper[d]["file_data"]+f"test/{cat}.json", 
                                                    prefix=prefix, device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                    with_knowledge=mapper[d]["with_knowledge"], max_seq=max_seq,
                                                    meta_type=mapper[d]["meta_type"])
                            shot_results[shots]["ppl"].append(ppl_score)
                        
                for shots, results in shot_results.items():
                    res_score = score(files_test=mapper[d]["file_data"]+f"test.json",files_to_score=results["generated_out"], meta_type="last_turn")
                    res_score["ppl"] = np.mean(results["ppl"])
                    print(res_score)
                    if checker_file(f"{d}_{shots}_{model_checkpoint}_{beam}-{args.do_sample}_{id_prefix}.json"):
                        save_file(f"{d}_{shots}_{model_checkpoint}_{beam}-{args.do_sample}_{id_prefix}.json", {"score":res_score,"generation":results["generated_out"]})
        else:
            prefix_list = load_prefix(tokenizer=tokenizer, shots_value=mapper[d]["shots"][max_seq], 
                                    shot_converter=mapper[d]["shot_converter"], 
                                    file_shot=mapper[d]["file_data"]+"valid.json", 
                                    name_dataset=d, with_knowledge=mapper[d]["with_knowledge"], 
                                    shot_separator=mapper[d]["shot_separator"],sample_times=args.sample_times)
        
            first_time = True
            for id_prefix, prefix_shots in enumerate(prefix_list):
                for shots, prefix in prefix_shots.items():
                    if shots == 0 and not first_time: continue 
                    first_time = False

                    if checker_file(f"{d}_{shots}_{model_checkpoint}_{beam}-{args.do_sample}_{id_prefix}.json") or args.verbose:
                        generation_out = generate_response(model, tokenizer, shot_converter=mapper[d]["shot_converter_inference"], 
                                                        file_to_eval=mapper[d]["file_data"]+"test.json", prefix=prefix, 
                                                        device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                        with_knowledge=mapper[d]["with_knowledge"], 
                                                        meta_type=mapper[d]["meta_type"], gen_len=mapper[d]["gen_len"], 
                                                        beam=beam, max_seq=max_seq, eos_token_id=198, 
                                                        do_sample=args.do_sample, multigpu=args.multigpu,verbose=args.verbose)

                        res_score = score(files_test=mapper[d]["file_data"]+"test.json",files_to_score=generation_out, meta_type=mapper[d]["meta_type"])
                        ppl_score = evalute_ppl(model, tokenizer, shot_converter=mapper[d]["shot_converter_inference"], 
                                                file_to_eval=mapper[d]["file_data"]+"test.json", 
                                                prefix=prefix, device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                with_knowledge=mapper[d]["with_knowledge"], max_seq=max_seq,
                                                meta_type=mapper[d]["meta_type"], verbose=args.verbose)
                        res_score["ppl"] = ppl_score
                        print(res_score)
                        save_file(f"{d}_{shots}_{model_checkpoint}_{beam}-{args.do_sample}_{id_prefix}.json", {"score":res_score,"generation":generation_out})

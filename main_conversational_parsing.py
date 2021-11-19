import json
import os
import argparse
from prompts.generic_prompt_parser import load_prefix, evalute_ppl, generate_response, generate_response_DKG
from prompts.dialKG_parser import convert_sample_to_shot_dialKG
from prompts.wizard_of_internet_parser import convert_sample_to_shot_wit
from prompts.wizard_of_wikipedia_parse import convert_sample_to_shot_wow
from prompts.semantic_parser import convert_sample_to_shot_semantic_parser
from prompts.mwoz_parser import convert_sample_to_shot_mwoz
from prompts.persona_parser import convert_sample_to_shot_msc
from tabulate import tabulate
from metric.scorer_parse import score
from py2neo import Graph
from utils.utils import load_model, save_file, checker_file
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


mapper = {
          "dialKG-parse": {"shot_converter":convert_sample_to_shot_dialKG, 
                 "file_data":"data/dialKG/parse-","level":"dialogue",
                  "shots":{1024:[1,2,3],2048:[1, 5, 10]},"shot_separator":"\n\n",
                  "meta_type":"sentence","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-hotel": {"shot_converter":convert_sample_to_shot_mwoz, 
                 "file_data":"data/mwoz/hotel-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-taxi": {"shot_converter":convert_sample_to_shot_mwoz, 
                 "file_data":"data/mwoz/taxi-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-train": {"shot_converter":convert_sample_to_shot_mwoz, 
                 "file_data":"data/mwoz/train-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-attraction": {"shot_converter":convert_sample_to_shot_mwoz, 
                 "file_data":"data/mwoz/attraction-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "mwoz-parse-dialogue-restaurant": {"shot_converter":convert_sample_to_shot_mwoz, 
                 "file_data":"data/mwoz/restaurant-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3, 5]},"shot_separator":"\n\n",
                  "meta_type":"predict","gen_len":50,"max_number_turns":5},
          "msc-parse-dialogue-1": {"shot_converter":convert_sample_to_shot_msc, 
                 "file_data":"data/msc/parse-session-1-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3]},"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":2},
          "msc-parse-dialogue-2": {"shot_converter":convert_sample_to_shot_msc, 
                 "file_data":"data/msc/parse-session-2-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3]},"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":2},
          "msc-parse-dialogue-3": {"shot_converter":convert_sample_to_shot_msc, 
                 "file_data":"data/msc/parse-session-3-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3]},"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":2},
          "msc-parse-dialogue-4": {"shot_converter":convert_sample_to_shot_msc, 
                 "file_data":"data/msc/parse-session-4-","level":"dialogue",
                  "shots":{1024:[0,1],2048:[0, 1, 3]},"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":2},
          "wit-parse": {"shot_converter":convert_sample_to_shot_wit, 
                 "file_data":"data/wit/parse-","level":"dialogue",
                  "shots":{1024:[0,1,5],2048:[0, 1, 5, 8]},"shot_separator":"\n\n",
                  "meta_type":"last_turn","gen_len":50,"max_number_turns":2},
          "wow-parse": {"shot_converter":convert_sample_to_shot_wow, 
                 "file_data":"data/wow/parse-","level":"dialogue",
                  "shots":{1024:[0, 1, 5],2048:[0, 1, 5, 10]},"shot_separator":"\n\n",
                  "meta_type":"last_turn","gen_len":50,"max_number_turns":2},
           "top": {"shot_converter":convert_sample_to_shot_semantic_parser, 
                    "shot_converter_inference": convert_sample_to_shot_semantic_parser,
                 "file_data":"data/TOP/","level":"sentence","level":None,
                  "shots":{1024:[1],2048:[1,10,25]},"shot_separator":"\n\n",
                  "meta_type":"sentence","gen_len":100,"max_number_turns":2},
           "semflow": {"shot_converter":convert_sample_to_shot_semantic_parser, 
                    "shot_converter_inference": convert_sample_to_shot_semantic_parser,
                 "file_data":"data/semflow/","level":"sentence","level":None,
                  "shots":{1024:[1],2048:[1,5,10]},"shot_separator":"\n\n",
                  "meta_type":"sentence","gen_len":100,"max_number_turns":2},
          "flowMWOZ": {"shot_converter":convert_sample_to_shot_semantic_parser, 
                    "shot_converter_inference": convert_sample_to_shot_semantic_parser,
                 "file_data":"data/flowMWOZ/","level":"sentence","level":None,
                  "shots":{1024:[1],2048:[1,5,10]},"shot_separator":"\n\n",
                  "meta_type":"sentence","gen_len":100,"max_number_turns":2}
         }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", default="gpt2",type=str,required=True)
    parser.add_argument("--dataset", default="persona",type=str,required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--sample_times", type=int, default=2)
    parser.add_argument("--do_sample", action='store_true', help="sample n times and rescore based on ppl")
    parser.add_argument("--multigpu", action='store_true', help="run on multiple gpus")
    parser.add_argument("--verbose", action='store_true', help="run on multiple gpus")
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    beam = args.beam
    model_checkpoint = args.model_checkpoint

    model, tokenizer, max_seq = load_model(args,model_checkpoint,device)
    
    list_of_dataset = ["persona", "wow", "ed"] if args.dataset == "all" else args.dataset.split(",")
    for d in list_of_dataset:
        print(f"EVALUATING DATASET {d} on {model_checkpoint} with beam size {beam}")
        prefix_list = load_prefix(tokenizer=tokenizer, shots_value=mapper[d]["shots"][max_seq], 
                                  shot_converter=mapper[d]["shot_converter"], 
                                  file_shot=mapper[d]["file_data"]+"valid.json", 
                                  name_dataset=d, level=mapper[d]["level"], 
                                  shot_separator=mapper[d]["shot_separator"],sample_times=args.sample_times)
        
        first_time = True
        for id_prefix, prefix_shots in enumerate(prefix_list):
            for shots, prefix in prefix_shots.items():
                if shots == 0 and not first_time: continue 
                first_time = False

                if checker_file(f"{d}_{shots}_{model_checkpoint}_{beam}-{args.do_sample}_{id_prefix}.json"):
                    if d == "dialKG-parse":
                        kg = Graph("http://eez114.ece.ust.hk:7474", auth=("neo4j", "CAiRE2020neo4j")) # Graph("ADDRESS", auth=("USR", "PWD"))
                        ### THIS REQUIRE A NEO4J DB UP and RUNNING
                        generation_out = generate_response_DKG(model, tokenizer, shot_converter=mapper[d]["shot_converter"], 
                                                    file_to_eval=mapper[d]["file_data"]+"test.json", prefix=prefix, 
                                                    device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                    level=mapper[d]["level"], 
                                                    meta_type=mapper[d]["meta_type"], gen_len=mapper[d]["gen_len"], 
                                                    beam=beam, max_seq=max_seq, eos_token_id=198, 
                                                    do_sample=args.do_sample, multigpu=args.multigpu, 
                                                    verbose=args.verbose, KG=kg)
                    else:
                        generation_out = generate_response(model, tokenizer, shot_converter=mapper[d]["shot_converter"], 
                                                        file_to_eval=mapper[d]["file_data"]+"test.json", prefix=prefix, 
                                                        device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                        level=mapper[d]["level"], 
                                                        meta_type=mapper[d]["meta_type"], gen_len=mapper[d]["gen_len"], 
                                                        beam=beam, max_seq=max_seq, eos_token_id=198, 
                                                        do_sample=args.do_sample, multigpu=args.multigpu, verbose=args.verbose)

                    res_score = score(files_test=mapper[d]["file_data"]+"test.json",files_to_score=generation_out, meta_type=d)
                    print(res_score)
                    ppl_score = evalute_ppl(model, tokenizer, shot_converter=mapper[d]["shot_converter"], 
                                            file_to_eval=mapper[d]["file_data"]+"test.json", 
                                            prefix=prefix, device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                            level=mapper[d]["level"], max_seq=max_seq,
                                            meta_type=mapper[d]["meta_type"], verbose=args.verbose)
                    res_score["ppl"] = ppl_score
                    print(res_score)
                    save_file(f"{d}_{shots}_{model_checkpoint}_{beam}-{args.do_sample}_{id_prefix}.json", {"score":res_score,"generation":generation_out})

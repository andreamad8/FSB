import os
import argparse
from prompts.generic_prompt_parser import evalute_ppl, generate_response
from prompts.semantic_parser import convert_sample_to_shot_semantic_parser
from prompts.wizard_of_wikipedia_parse import convert_sample_to_shot_wow
from prompts.dialKG_parser import convert_sample_to_shot_dialKG
from metric.scorer_parse import score
from py2neo import Graph
from utils.utils import load_model, save_file, checker_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"

mapper = {
           "top": {"shot_converter":convert_sample_to_shot_semantic_parser, 
                  "shots":{1024:[1, 10],2048:[25]},"shot_separator":"\n\n",
                  "meta_type":"sentencedynamic","gen_len":100,"max_number_turns":2},
           "semflow": {"shot_converter":convert_sample_to_shot_semantic_parser, 
                  "shots":{1024:[1, 10],2048:[10]},"shot_separator":"\n\n",
                  "meta_type":"sentencedynamic","gen_len":100,"max_number_turns":2},
           "flowMWOZ": {"shot_converter":convert_sample_to_shot_semantic_parser, 
                  "shots":{1024:[1, 10],2048:[10]},"shot_separator":"\n\n",
                  "meta_type":"sentencedynamic","gen_len":100,"max_number_turns":2},
           "wow-parse": {"shot_converter":convert_sample_to_shot_wow, 
                  "shots":{1024:[10],2048:[10]},"shot_separator":"\n\n",
                  "meta_type":"sentencedynamic","gen_len":50,"max_number_turns":2},
           "wit-parse": {"shot_converter":convert_sample_to_shot_wow, 
                  "shots":{1024:[10],2048:[8]},"shot_separator":"\n\n",
                  "meta_type":"sentencedynamic","gen_len":50,"max_number_turns":2},
           "dialKG-parse": {"shot_converter":convert_sample_to_shot_dialKG, 
                  "shots":{1024:[10],2048:[10]},"shot_separator":"\n\n",
                  "meta_type":"sentencedynamic","gen_len":50,"max_number_turns":2},
         }

if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument("--model_checkpoint", default="gpt2",type=str,required=True)
       parser.add_argument("--dataset", default="flowMWOZ",type=str,required=True)
       parser.add_argument("--filedata", default="data/flowMWOZ/test_dynamic.json",type=str,required=True)
       parser.add_argument("--gpu", type=int, default=-1)
       parser.add_argument("--beam", type=int, default=1)
       parser.add_argument("--sample_times", type=int, default=3)
       parser.add_argument("--do_sample", action='store_true', help="sample n times and rescore based on ppl")
       parser.add_argument("--multigpu", action='store_true', help="run on multiple gpus")
       parser.add_argument("--verbose", action='store_true', help="run on multiple gpus")
       args = parser.parse_args()

       if args.gpu >=0:
              device = f'cuda:{args.gpu}'
       else:
              device = "cpu"
       beam = args.beam
       model_checkpoint = args.model_checkpoint

       model, tokenizer, max_seq = load_model(args,model_checkpoint,device)
       
       list_of_dataset = args.dataset.split(",")
       d = list_of_dataset[0]

       print(f"EVALUATING DATASET {d} on {model_checkpoint} with beam size {beam}")
       name_experiment = args.filedata.split("test_dynamic_")[-1].replace(".json","")
       print(name_experiment)
       first_time = True
       for shots in mapper[d]["shots"][max_seq]:
              if shots == 0 and not first_time: continue 
              first_time = False
              print(f"RUNNING {shots}")
              if checker_file(f"{d}_{shots}_{model_checkpoint}_{name_experiment}.json") or args.verbose:

                     if d == "dialKG-parse":
                            from prompts.generic_prompt_parser import generate_response_DKG, evalute_ppl
                            from metric.scorer_parse import score


                            kg = Graph("http://eez114.ece.ust.hk:7474", auth=("neo4j", "CAiRE2020neo4j")) # Graph("ADDRESS", auth=("USR", "PWD"))
                            ### THIS REQUIRE A NEO4J DB UP and RUNNING
                            generation_out = generate_response_DKG(model, tokenizer, shot_converter=mapper[d]["shot_converter"], 
                                                        file_to_eval=args.filedata, prefix=shots, 
                                                        device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                        level=None, 
                                                        meta_type=mapper[d]["meta_type"], gen_len=mapper[d]["gen_len"], 
                                                        beam=beam, max_seq=max_seq, eos_token_id=198, 
                                                        do_sample=args.do_sample, multigpu=args.multigpu, 
                                                        verbose=args.verbose, KG=kg)

                            res_score = score(files_test=args.filedata,files_to_score=generation_out, meta_type=d)
                            print(res_score)

                            ppl_score = evalute_ppl(model, tokenizer, shot_converter=mapper[d]["shot_converter"], 
                                                        file_to_eval=args.filedata, 
                                                        prefix=shots, device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                        level=None, max_seq=max_seq,
                                                        meta_type=mapper[d]["meta_type"], verbose=args.verbose)
                            res_score["ppl"] = ppl_score
                            print(res_score)
                            save_file(f"{d}_{shots}_{model_checkpoint}_{name_experiment}.json", {"score":res_score,"generation":generation_out})

                     else:
                            generation_out = generate_response(model, tokenizer, shot_converter=mapper[d]["shot_converter"], 
                                                               file_to_eval=args.filedata, prefix=shots, 
                                                               device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                               level=None, 
                                                               meta_type=mapper[d]["meta_type"], gen_len=mapper[d]["gen_len"], 
                                                               beam=beam, max_seq=max_seq, eos_token_id=198, 
                                                               do_sample=args.do_sample, multigpu=args.multigpu,verbose=args.verbose)

                            res_score = score(files_test=args.filedata,files_to_score=generation_out, meta_type=d)
                            print(res_score)

                            ppl_score = evalute_ppl(model, tokenizer, shot_converter=mapper[d]["shot_converter"], 
                                                        file_to_eval=args.filedata, 
                                                        prefix=shots, device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                        level=None, max_seq=max_seq,
                                                        meta_type=mapper[d]["meta_type"], verbose=args.verbose)
                            res_score["ppl"] = ppl_score
                            print(res_score)
                            save_file(f"{d}_{shots}_{model_checkpoint}_{name_experiment}.json", {"score":res_score,"generation":generation_out})

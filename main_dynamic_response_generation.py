import os
import argparse
from prompts.generic_prompt import evalute_ppl, generate_response_dynamic
from prompts.bAbi_dialogue import convert_sample_to_shot_bAbi
from metric.scorer_parse import score
from utils.utils import load_model, save_file, checker_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"

mapper = {
          "babi5": {"shot_converter":convert_sample_to_shot_bAbi, 
                    "shot_converter_inference": convert_sample_to_shot_bAbi,
                     "file_data":"data/dialog-bAbI-tasks/bAbI-dial-5-","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[1,0,1,2]},"shot_separator":"\n\n",
                     "meta_type":"alldynamic","gen_len":50,"max_number_turns":3},
          "babi5-OOV": {"shot_converter":convert_sample_to_shot_bAbi, 
                    "shot_converter_inference": convert_sample_to_shot_bAbi,
                     "file_data":"data/dialog-bAbI-tasks/bAbI-dial-5-OOV-","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2]},"shot_separator":"\n\n",
                     "meta_type":"alldynamic","gen_len":50,"max_number_turns":3},
          "babi6": {"shot_converter":convert_sample_to_shot_bAbi, 
                    "shot_converter_inference": convert_sample_to_shot_bAbi,
                     "file_data":"data/dialog-bAbI-tasks/bAbI-dial-6-","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2]},"shot_separator":"\n\n",
                     "meta_type":"alldynamic","gen_len":50,"max_number_turns":3},
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


                     generation_out = generate_response_dynamic(model, tokenizer, shot_converter=mapper[d]["shot_converter"], 
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

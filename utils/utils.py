import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer 
import json
import os

def load_model(args,model_checkpoint,device):
    print(f"LOADING {model_checkpoint}")
    if "gpt-j"in model_checkpoint or "neo"in model_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        if args.multigpu:
            from parallelformers import parallelize
            parallelize(model, num_gpus=4, fp16=True, verbose='detail')
        else:
            model.half().to(device)
        max_seq = 2048
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        tokenizer.bos_token = ":"
        tokenizer.eos_token = "\n"
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        max_seq = 1024
        model.half().to(device)
    print("DONE LOADING")
    
    return model, tokenizer, max_seq


def save_file(filename, results):
    filename = filename.replace("EleutherAI/","")
    with open(f'generations/{filename}', 'w') as fp:
        json.dump(results, fp, indent=4)

def checker_file(filename):
    filename = filename.replace("EleutherAI/","")
    if os.path.exists(f'generations/{filename}'):
        print(f"generations/{filename} already exists! ==> Skipping the file" )
    return not os.path.exists(f'generations/{filename}')

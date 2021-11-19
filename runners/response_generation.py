from joblib import Parallel, delayed
import queue
import os
import time

# Define number of GPUs available
GPU_available = [0,1,2,3,4,5,6,7,8]
N_GPU = len(GPU_available)

models = ["gpt2","gpt2-medium","gpt2-large","gpt2-xl",
          "EleutherAI/gpt-neo-1.3B","EleutherAI/gpt-neo-2.7B",
          "EleutherAI/gpt-j-6B"]
datasets = ["persona","smd-navigate","smd-schedule",
            "smd-weather","msc-dialogue-2","wow",
            "wit","ed","dialKG","DD","IC","IC-img","coQA"]

template = "python main_response_generation.py --model_checkpoint {} --dataset {} --gpu "
experiments = []
for m in models: 
    for d in datasets:
        experiments.append(template.format(m,d))


# Put indices in queue
q = queue.Queue(maxsize=N_GPU)
mapper = {}
invert_mapper = {}
for i in range(N_GPU):
    mapper[i] = GPU_available[i]
    invert_mapper[GPU_available[i]] = i
    q.put(i)

def runner(cmd):
    gpu = mapper[q.get()]
    print("RUNNING: ",str(cmd)+str(gpu))
    os.system(str(cmd)+str(gpu))
    q.put(invert_mapper[gpu])

# Change loop
Parallel(n_jobs=N_GPU, backend="threading")( delayed(runner)(e) for e in experiments)


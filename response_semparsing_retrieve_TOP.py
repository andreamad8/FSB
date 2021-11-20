from joblib import Parallel, delayed
import queue
import os
import time

# Define number of GPUs available
GPU_available = [0]
N_GPU = len(GPU_available)

models = ["EleutherAI/gpt-j-6B"]
datasets = ["top"]
files = [
        "data/TOP/test_dynamic_all-mpnet-base-v2_0.01_0.json",
        "data/TOP/test_dynamic_all-mpnet-base-v2_0.1_0.json",
        "data/TOP/test_dynamic_all-mpnet-base-v2_0.25_0.json",
        "data/TOP/test_dynamic_all-mpnet-base-v2_1.0_0.json",
        "data/TOP/test_dynamic_tfidf_0.01_0.json",
        "data/TOP/test_dynamic_tfidf_0.1_0.json",
        "data/TOP/test_dynamic_tfidf_0.25_0.json",
        "data/TOP/test_dynamic_tfidf_1.0_0.json"]

template = "python main_dynamic_generation.py --model_checkpoint {} --dataset {} --filedata {} --gpu "
experiments = []
for m in models: 
    for d in datasets:
        for f in files:
            experiments.append(template.format(m,d,f))


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